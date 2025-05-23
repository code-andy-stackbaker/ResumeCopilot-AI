import logging
import os
import sys
from datetime import datetime, timedelta

# Airflow imports
from airflow.models import DAG # type: ignore
from airflow.operators.dummy import DummyOperator # type: ignore
from airflow.operators.python import PythonOperator # type: ignore
from airflow.utils.dates import days_ago # type: ignore


# --- Add project root to sys.path for Airflow to find custom modules ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

log = logging.getLogger(__name__)
log.info(f"DAG: Project root set to: {PROJECT_ROOT}. Added to sys.path for imports.")

try:
    from app.database.session import SessionLocal 
    from app.database.models import Job # For type hinting, table access etc.
    # from app.config import SOME_CONFIG_IF_NEEDED_DIRECTLY_IN_DAG 
    
    # Updated import: We only need generate_embeddings_for_texts from embedding_utils directly in the DAG tasks now
    from utils.embedding_utils import generate_embeddings_for_texts 
    from utils.db_utils import fetch_jobs_missing_embeddings_sqlalchemy, update_job_embeddings_batch_sqlalchemy
except ImportError as e:
    log.error(f"DAG IMPORT ERROR: Critical modules not found. Check PROJECT_ROOT ('{PROJECT_ROOT}') "
              f"and ensure '__init__.py' exists in subdirectories like 'utils/' and 'app/'. Error: {e}")
    log.error(f"Current sys.path for DAG: {sys.path}")
    raise 

# --- Global Logging Setup (app.config should have initialized basicConfig) ---
# log.setLevel(logging.INFO) # Assuming basicConfig is set by app.config or entrypoint

# Configuration for this DAG run
DB_FETCH_BATCH_SIZE = int(os.getenv("DB_FETCH_BATCH_SIZE", "100")) # Example: Make it configurable via env var

# --- Python Callable Functions for DAG Tasks ---

def _fetch_jobs_needing_embeddings_callable(**kwargs):
    ti = kwargs.get('ti')
    log.info(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - Starting (batch size: {DB_FETCH_BATCH_SIZE}).")
    
    db_session = None
    jobs_to_process_for_xcom = []
    try:
        db_session = SessionLocal()
        log.info("DAG TASK: _fetch_jobs_needing_embeddings_callable - SQLAlchemy session created.")
        
        job_model_instances = fetch_jobs_missing_embeddings_sqlalchemy(
            db_session=db_session, 
            batch_size=DB_FETCH_BATCH_SIZE
        )
        log.info(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - number of jobs fetched from db: {len(jobs_to_process_for_xcom)} job(s).")
        if job_model_instances:
            for job_instance in job_model_instances:
                jobs_to_process_for_xcom.append({
                    'job_pk_id': job_instance.id,
                    'external_job_id': job_instance.external_job_id, 
                    'job_description_text': job_instance.job_description_text
                })
            log.info(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - Fetched {len(jobs_to_process_for_xcom)} job(s).")
        else:
            log.info("DAG TASK: _fetch_jobs_needing_embeddings_callable - No new jobs found needing embeddings.")
            
    except Exception as e:
        log.error(f"DAG TASK: _fetch_jobs_needing_embeddings_callable - Error: {e}", exc_info=True)
        raise 
    finally:
        if db_session:
            db_session.close()
            log.info("DAG TASK: _fetch_jobs_needing_embeddings_callable - SQLAlchemy session closed.")
            
    if ti and jobs_to_process_for_xcom: # Push only if there's data to avoid empty XComs if not handled by receiver
        ti.xcom_push(key='jobs_for_embedding', value=jobs_to_process_for_xcom)
        ti.xcom_push(key='jobs_for_embedding_count', value=len(jobs_to_process_for_xcom))
    elif ti: # still push count if empty
        ti.xcom_push(key='jobs_for_embedding_count', value=0)
        # Optionally push an empty list for 'jobs_for_embedding' if downstream tasks expect the key
        # ti.xcom_push(key='jobs_for_embedding', value=[])
        
    # The function returns the list of jobs, which Airflow automatically pushes to XComs under 'return_value'
    # if no explicit ti.xcom_push for the main data payload is made.
    # To be explicit and control the key name, using ti.xcom_push(key='jobs_for_embedding', ...) is better.
    # For now, let's rely on the explicit push above and return the count for direct logging/observability if needed.
    return len(jobs_to_process_for_xcom) # Or return jobs_to_process_for_xcom if preferred for 'return_value'


def _generate_embeddings_for_db_jobs_callable(**kwargs):
    ti = kwargs['ti']
    # Pulling the explicitly keyed XCom value
    jobs_to_embed_dicts = ti.xcom_pull(task_ids='fetch_jobs_needing_embeddings_task', key='jobs_for_embedding') 
    
    if not jobs_to_embed_dicts:
        log.info("DAG TASK: _generate_embeddings_for_db_jobs_callable - No job data received. Skipping.")
        if ti:
            ti.xcom_push(key='embeddings_generated_count', value=0)
            ti.xcom_push(key='job_updates_payload', value=[]) # Push empty list for downstream
        return 0 # Or []

    log.info(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Received {len(jobs_to_embed_dicts)} entries.")
    
    # Filter out entries with missing or empty job descriptions before sending to embedding model
    valid_jobs_for_embedding = []
    descriptions_to_embed = []
    for job_dict in jobs_to_embed_dicts:
        if job_dict.get('job_description_text') and isinstance(job_dict['job_description_text'], str) and job_dict['job_description_text'].strip():
            valid_jobs_for_embedding.append(job_dict)
            descriptions_to_embed.append(job_dict['job_description_text'])
        else:
            log.warning(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Skipping job PK ID "
                        f"{job_dict.get('job_pk_id', 'Unknown')} due to missing/empty description.")

    if not descriptions_to_embed:
        log.info("DAG TASK: _generate_embeddings_for_db_jobs_callable - No valid job descriptions found to embed.")
        if ti:
            ti.xcom_push(key='embeddings_generated_count', value=0)
            ti.xcom_push(key='job_updates_payload', value=[])
        return 0 # Or []

    # generate_embeddings_for_texts now handles model loading/caching internally
    # and returns list[list[float]]
    try:
        embedding_vectors_list = generate_embeddings_for_texts(texts=descriptions_to_embed)
    except Exception as e:
        log.error(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Error during embedding generation: {e}", exc_info=True)
        raise # Fail the task if embedding generation fails

    if len(embedding_vectors_list) != len(descriptions_to_embed):
        error_msg = (f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Mismatch: "
                     f"{len(descriptions_to_embed)} descriptions, {len(embedding_vectors_list)} embeddings generated.")
        log.error(error_msg)
        # Depending on severity, you might want to raise an error or try to proceed with partial data.
        # For now, let's raise if there's a mismatch after successful generation call.
        raise ValueError(error_msg)

    job_updates_payload = []
    for i, job_dict_original in enumerate(valid_jobs_for_embedding):
        # Assuming embedding_vectors_list directly corresponds to descriptions_to_embed order
        if i < len(embedding_vectors_list):
            job_updates_payload.append({
                'id': job_dict_original['job_pk_id'], 
                'external_job_id': job_dict_original['external_job_id'], 
                'embedding_vector': embedding_vectors_list[i] # Already list[float]
            })
        else: # Should not happen if previous length check is robust
            log.error(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Index out of bounds for job PK ID {job_dict_original['job_pk_id']}. This indicates a logic flaw.")

    log.info(f"DAG TASK: _generate_embeddings_for_db_jobs_callable - Prepared {len(job_updates_payload)} updates.")
    if ti:
        ti.xcom_push(key='job_updates_payload', value=job_updates_payload)
        ti.xcom_push(key='embeddings_generated_count', value=len(job_updates_payload))
    return len(job_updates_payload) # Or job_updates_payload


def _update_job_embeddings_in_db_callable(**kwargs):
    ti = kwargs['ti']
    # Pulling the explicitly keyed XCom value
    job_updates_list = ti.xcom_pull(task_ids='generate_embeddings_for_db_jobs_task', key='job_updates_payload')
    
    if not job_updates_list:
        log.info("DAG TASK: _update_job_embeddings_in_db_callable - No updates received. Nothing to process.")
        if ti:
            ti.xcom_push(key='db_updated_count', value=0)
        return 0

    log.info(f"DAG TASK: _update_job_embeddings_in_db_callable - Received {len(job_updates_list)} job(s) to update.")
    
    db_session = None
    updated_count = 0
    try:
        db_session = SessionLocal()
        log.info("DAG TASK: _update_job_embeddings_in_db_callable - SQLAlchemy session created.")
        
        updated_count = update_job_embeddings_batch_sqlalchemy(
            db_session=db_session, 
            job_updates=job_updates_list # Expects list of dicts with 'id' and 'embedding_vector'
        )
        
        if updated_count > 0:
            db_session.commit()
            log.info(f"DAG TASK: _update_job_embeddings_in_db_callable - SQLAlchemy session committed. "
                     f"Updated embeddings for {updated_count} job(s).")
        elif updated_count == 0 and job_updates_list: # Received data but util updated none
             log.info("DAG TASK: _update_job_embeddings_in_db_callable - Utility function reported 0 updates, though data was passed. Session not committed.")
        else: # No updates to make or reported by util
            log.info("DAG TASK: _update_job_embeddings_in_db_callable - No actual updates made by utility. Session not committed.")

    except Exception as e:
        log.error(f"DAG TASK: _update_job_embeddings_in_db_callable - Error during DB update: {e}", exc_info=True)
        if db_session:
            log.info("DAG TASK: _update_job_embeddings_in_db_callable - Rolling back SQLAlchemy session.")
            db_session.rollback()
        raise 
    finally:
        if db_session:
            db_session.close()
            log.info("DAG TASK: _update_job_embeddings_in_db_callable - SQLAlchemy session closed.")
            
    if ti:
        ti.xcom_push(key='db_updated_count', value=updated_count)
    return updated_count

# --- Default DAG Arguments ---
default_args = {
    'owner': 'Adeel - MLOps Sprint Day 1 DB Backfill', # Persona specific owner
    'depends_on_past': False,
    'start_date': days_ago(0), 
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1, 
    'retry_delay': timedelta(minutes=3),
}

# # --- DAG Definition ---
# with DAG(
#     dag_id='adeel_db_embedding_backfill_v1', 
#     default_args=default_args,
#     description='Periodically checks DB for jobs missing embeddings, generates, and updates them using pgvector.',
#     schedule_interval='@hourly', 
#     catchup=False,
#     max_active_runs=1, 
#     tags=['mlops_sprint_adeel', 'day1', 'db_backfill', 'embeddings', 'pgvector'],
# ) as dag:

    # --- DAG Definition ---
with DAG(
    dag_id='adeel_db_embedding_backfill_v1',
    default_args=default_args,
    description='Periodically checks DB for jobs missing embeddings, generates, and updates them using pgvector.',
    # MODIFICATION HERE: Run every 2 minutes
    schedule_interval='*/2 * * * *', # Cron expression for every 2 minutes
    catchup=False, # IMPORTANT: This ensures it only runs for the latest interval
    max_active_runs=1,
    tags=['mlops_sprint_adeel', 'day1', 'db_backfill', 'embeddings', 'pgvector'],
) as dag:
    # ... your tasks (start_pipeline, fetch_jobs_needing_embeddings_task, etc.) ...

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