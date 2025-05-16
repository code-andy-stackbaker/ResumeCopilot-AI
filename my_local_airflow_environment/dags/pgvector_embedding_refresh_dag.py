import logging
import os
import sys
from datetime import datetime, timedelta

import airflow # type: ignore
from airflow.models import DAG # type: ignore
from airflow.operators.dummy import DummyOperator # type: ignore
from airflow.operators.python import PythonOperator # type: ignore

# --- Add project root to sys.path for Airflow to find custom modules ---
# This DAG file is expected to be in PROJECT_ROOT/airflow_docker_env/dags/
# So, PROJECT_ROOT is two levels up from os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

log = logging.getLogger(__name__) # Define log early for robust logging
log.info(f"DAG: Project root set to: {PROJECT_ROOT}. Added to sys.path for imports.")

try:
    # Import SQLAlchemy SessionLocal for creating DB sessions within tasks
    from app.database.session import SessionLocal 
    # Import your Job model for type hinting and to access table name, etc.
    from app.database.models import Job
    # Import configurations (though utils mostly use them directly)
    # from app.config import SOME_CONFIG_IF_NEEDED_DIRECTLY_IN_DAG 
    
    # Import your utility functions
    from utils.embedding_utils import generate_embeddings_for_texts, get_sentence_embedding_model
    from utils.db_utils import fetch_jobs_missing_embeddings_sqlalchemy, update_job_embeddings_batch_sqlalchemy
except ImportError as e:
    log.error(f"DAG IMPORT ERROR: Critical modules not found. Check PROJECT_ROOT ('{PROJECT_ROOT}') "
              f"and ensure '__init__.py' exists in 'PROJECT_ROOT/utils/' "
              f"and all necessary files are present. Error: {e}")
    log.error(f"Current sys.path for DAG: {sys.path}")
    raise # Fail fast if imports are broken

# --- Global Logging Setup (app.config should have initialized basicConfig) ---
log.setLevel(logging.INFO)

# Configuration for this DAG run
DB_FETCH_BATCH_SIZE = 100 # Number of jobs to process per DAG run

# --- Python Callable Functions for DAG Tasks ---

def _fetch_jobs_needing_embeddings_callable(**kwargs):
    """
    Fetches a batch of job records from the database that are missing embeddings.
    Returns a list of dictionaries, each containing 'job_pk_id', 'external_job_id', 
    and 'job_description_text' for XComs.
    """
    ti = kwargs.get('ti')
    log.info(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - Starting to fetch jobs (batch size: {DB_FETCH_BATCH_SIZE})...")
    
    db_session = None
    jobs_to_process_for_xcom = []
    try:
        db_session = SessionLocal()
        log.info("DAG TASK: _fetch_jobs_needing_embeddings_callable - SQLAlchemy session created.")
        
        # Fetch Job model instances
        job_model_instances = fetch_jobs_missing_embeddings_sqlalchemy(
            db_session=db_session, 
            batch_size=DB_FETCH_BATCH_SIZE
        )
        
        if job_model_instances:
            # Convert to a list of simple dicts for XCom compatibility and to pass only needed data
            for job_instance in job_model_instances:
                jobs_to_process_for_xcom.append({
                    'job_pk_id': job_instance.id, # Primary Key
                    'external_job_id': job_instance.external_job_id, # Business Key
                    'job_description_text': job_instance.job_description_text
                })
            log.info(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - Fetched {len(jobs_to_process_for_xcom)} job(s) needing embedding.")
        else:
            log.info("DAG TASK: _fetch_jobs_needing_embeddings_callable - No jobs found needing embeddings in this batch.")
            
    except Exception as e:
        log.error(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - Error: {e}", exc_info=True)
        raise # Fail the task
    finally:
        if db_session:
            db_session.close()
            log.info("DAG TASK: _fetch_jobs_needing_embeddings_callable - SQLAlchemy session closed.")
            
    if ti:
        ti.xcom_push(key='jobs_for_embedding_count', value=len(jobs_to_process_for_xcom))
    return jobs_to_process_for_xcom # List of dicts


def _generate_embeddings_for_db_jobs_callable(**kwargs):
    """
    Generates embeddings for job descriptions pulled from XComs.
    Returns a list of dicts, each with job identifiers and the embedding vector.
    """
    ti = kwargs['ti']
    jobs_to_embed_dicts = ti.xcom_pull(task_ids='fetch_jobs_needing_embeddings_task', key='return_value')
    
    if not jobs_to_embed_dicts:
        log.info("DAG TASK: _generate_embeddings_for_db_jobs_callable - No job data received. Skipping embedding generation.")
        if ti:
            ti.xcom_push(key='embeddings_generated_count', value=0)
        return []

    log.info(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Received {len(jobs_to_embed_dicts)} job entries for embedding.")
    
    descriptions = [job_dict['job_description_text'] for job_dict in jobs_to_embed_dicts if job_dict.get('job_description_text')]
    
    if not descriptions:
        log.info("DAG TASK: _generate_embeddings_for_db_jobs_callable - No valid descriptions found in job data.")
        if ti:
            ti.xcom_push(key='embeddings_generated_count', value=0)
        return []

    # Generate embeddings (model name comes from app.config via the util)
    # This utility function directly returns list[list[float]]
    embedding_vectors_list = generate_embeddings_for_texts(texts=descriptions)

    if len(embedding_vectors_list) != len(descriptions): # Should match count of non-empty descriptions
        error_msg = (f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Mismatch in embedding results. "
                     f"Processed {len(descriptions)} descriptions, but got {len(embedding_vectors_list)} embeddings.")
        log.error(error_msg)
        raise ValueError(error_msg)

    # Prepare data for the next task: list of dicts with job ID and its embedding
    job_updates_payload = []
    desc_idx = 0
    for job_dict_original in jobs_to_embed_dicts:
        if job_dict_original.get('job_description_text'): # If it was one that we sent for embedding
            if desc_idx < len(embedding_vectors_list):
                job_updates_payload.append({
                    'id': job_dict_original['job_pk_id'], # Pass the primary key for precise update
                    'external_job_id': job_dict_original['external_job_id'], # For logging/reference
                    'embedding_vector': embedding_vectors_list[desc_idx]
                })
                desc_idx +=1
            else:
                log.error(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Logic error: ran out of embeddings for job PK ID {job_dict_original['job_pk_id']}")
    
    log.info(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Prepared {len(job_updates_payload)} job updates with embeddings.")
    if ti:
        ti.xcom_push(key='embeddings_generated_count', value=len(job_updates_payload))
    return job_updates_payload


def _update_job_embeddings_in_db_callable(**kwargs):
    """
    Updates existing job records in the database with their newly generated embeddings.
    Manages the SQLAlchemy session for this batch of updates.
    """
    ti = kwargs['ti']
    job_updates_list = ti.xcom_pull(task_ids='generate_embeddings_for_db_jobs_task', key='return_value')
    
    if not job_updates_list:
        log.info("DAG TASK: _update_job_embeddings_in_db_callable - No job updates received. Nothing to process.")
        if ti:
            ti.xcom_push(key='db_updated_count', value=0)
        return

    log.info(f"DAG TASK: _update_job_embeddings_in_db_callable - Received {len(job_updates_list)} job(s) to update with embeddings.")
    
    db_session = None
    updated_count = 0
    try:
        db_session = SessionLocal()
        log.info("DAG TASK: _update_job_embeddings_in_db_callable - SQLAlchemy session created.")
        
        # Call the utility function to perform the batch update logic
        updated_count = update_job_embeddings_batch_sqlalchemy(
            db_session=db_session, 
            job_updates=job_updates_list
        )
        
        if updated_count > 0:
            db_session.commit()
            log.info(f"DAG TASK: _update_job_embeddings_in_db_callable - SQLAlchemy session committed. "
                     f"Successfully updated embeddings for {updated_count} job(s).")
        else:
            log.info("DAG TASK: _update_job_embeddings_in_db_callable - No jobs were actually updated by the utility function (session not committed).")
            # No commit needed if util function did no work or returned 0 updates prepared

    except Exception as e:
        log.error(f"DAG TASK: _update_job_embeddings_in_db_callable - Error during DB update process: {e}", exc_info=True)
        if db_session:
            log.info("DAG TASK: _update_job_embeddings_in_db_callable - Rolling back SQLAlchemy session due to error.")
            db_session.rollback()
        raise # Re-raise to mark Airflow task as failed
    finally:
        if db_session:
            db_session.close()
            log.info("DAG TASK: _update_job_embeddings_in_db_callable - SQLAlchemy session closed.")
            
    if ti:
        ti.xcom_push(key='db_updated_count', value=updated_count)


# --- Default DAG Arguments ---
default_args = {
    'owner': 'Adeel - MLOps Sprint Day 1 DB Backfill',
    'depends_on_past': False,
    'start_date': days_ago(0), # Consider days_ago(1) if you want it to run for "yesterday" on first unpause
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1, # Number of retries before failing task
    'retry_delay': timedelta(minutes=3), # How long to wait before retrying
}

# --- DAG Definition ---
with DAG(
    dag_id='adeel_db_embedding_backfill_v1', # New DAG ID reflecting its purpose
    default_args=default_args,
    description='Periodically checks DB for jobs missing embeddings, generates, and updates them.',
    schedule_interval='@hourly', # Or timedelta(hours=1), or as frequent as you need to check
    catchup=False,
    max_active_runs=1, # Prevent multiple DAG runs from overlapping if processing takes time
    tags=['mlops_sprint_adeel', 'day1', 'db_backfill', 'embeddings', 'pgvector'],
) as dag:

    start_pipeline = DummyOperator(
        task_id='start_pipeline',
    )

    fetch_jobs_needing_embeddings_task = PythonOperator(
        task_id='fetch_jobs_needing_embeddings_task',
        python_callable=_fetch_jobs_needing_embeddings_callable,
    )

    generate_embeddings_for_db_jobs_task = PythonOperator(
        task_id='generate_embeddings_for_db_jobs_task',
        python_callable=_generate_embeddings_for_db_jobs_callable,
    )

    update_job_embeddings_in_db_task = PythonOperator(
        task_id='update_job_embeddings_in_db_task',
        python_callable=_update_job_embeddings_in_db_callable,
    )

    end_pipeline = DummyOperator(
        task_id='end_pipeline',
    )

    # Define Task Dependencies
    start_pipeline >> fetch_jobs_needing_embeddings_task >> \
    generate_embeddings_for_db_jobs_task >> update_job_embeddings_in_db_task >> \
    end_pipeline