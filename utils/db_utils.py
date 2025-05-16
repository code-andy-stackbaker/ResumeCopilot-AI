# File: PROJECT_ROOT/utils/db_utils.py
import logging
import os
import sys
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd # For robust datetime parsing in last_updated_raw handling

# Attempt to import app modules. The primary caller (DAG or another script)
# should ensure PROJECT_ROOT is in sys.path.
try:
    from app.database.models import Job # Your SQLAlchemy Job model
    # SessionLocal is not directly used in this util file's functions,
    # as the session is passed in. It's imported by the calling code.
    # from app.config import EMBEDDING_DIMENSION_RECOMMENDER # Only if needed for validation here
except ImportError:
    # Fallback for trying to run/test this util standalone
    _PROJECT_ROOT_FOR_UTIL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT_FOR_UTIL not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_FOR_UTIL)
    try:
        from app.database.models import Job
        # from app.config import EMBEDDING_DIMENSION_RECOMMENDER
    except ImportError as e_inner:
        # Use a basic logger if app.config logger isn't available yet
        _temp_log = logging.getLogger(__name__)
        _temp_log.error(f"ERROR in db_utils.py: Could not import 'Job' model from app.database.models. "
                        f"Ensure PROJECT_ROOT is in sys.path. Attempted fallback path: {_PROJECT_ROOT_FOR_UTIL}. Error: {e_inner}")
        raise

log = logging.getLogger(__name__) # Uses the logger configured in app/config.py

def fetch_jobs_missing_embeddings_sqlalchemy(db_session: Session, batch_size: int = 100) -> list[Job]:
    log.info(f"DB_UTILS (SQLAlchemy): Fetching up to {batch_size} jobs missing embeddings...")
    try:
        jobs_needing_embedding = (
            db_session.query(Job)
            .filter(Job.job_description_embedding == None) # SQLAlchemy syntax for IS NULL
            .limit(batch_size)
            .all()
        )
        log.info(f"DB_UTILS (SQLAlchemy): Found {len(jobs_needing_embedding)} jobs missing embeddings.")
        return jobs_needing_embedding
    except Exception as e:
        log.error(f"DB_UTILS (SQLAlchemy): Error fetching jobs missing embeddings: {e}", exc_info=True)
        raise

def update_job_embeddings_batch_sqlalchemy(db_session: Session, job_updates: list[dict]) -> int:
    if not job_updates:
        log.info("DB_UTILS (SQLAlchemy): No job updates provided for batch.")
        return 0

    log.info(f"DB_UTILS (SQLAlchemy): Preparing to update embeddings for {len(job_updates)} jobs.")
    updated_count = 0
    for update_info in job_updates:
        job_pk_id = update_info.get('id') 
        external_id = update_info.get('external_job_id')
        embedding_vector = update_info.get('embedding_vector')

        if embedding_vector is None or not isinstance(embedding_vector, list):
            log.warning(f"DB_UTILS (SQLAlchemy): Skipping update for job (PK ID: {job_pk_id}, Ext. ID: {external_id}) "
                        "due to missing or invalid 'embedding_vector'.")
            continue

        try:
            job_to_update = None
            if job_pk_id is not None: # Prefer fetching by primary key if available
                job_to_update = db_session.query(Job).filter(Job.id == job_pk_id).first()
            elif external_id: # Fallback to external_job_id
                job_to_update = db_session.query(Job).filter(Job.external_job_id == external_id).first()
            else:
                log.warning("DB_UTILS (SQLAlchemy): Skipping update due to missing 'id' or 'external_job_id' in update_info.")
                continue
            
            if job_to_update:
                log.debug(f"DB_UTILS (SQLAlchemy): Preparing to update embedding for Job ID: {job_to_update.id} (External ID: {job_to_update.external_job_id})")
                job_to_update.job_description_embedding = embedding_vector # pgvector-sqlalchemy handles list assignment
                
                # If your Job model has `last_updated_ts` and `onupdate=func.now()`,
                # SQLAlchemy should handle updating it automatically when the session is flushed
                # because a change was made to job_description_embedding.
                # If you want to be explicit or your model doesn't have onupdate:
                if hasattr(job_to_update, 'last_updated_ts'):
                    job_to_update.last_updated_ts = datetime.now() # Or pass a specific timestamp if available
                
                updated_count += 1
            else:
                log.warning(f"DB_UTILS (SQLAlchemy): Job not found in DB for update. PK ID: {job_pk_id}, Ext. ID: {external_id}")
        
        except Exception as e:
            log.error(f"DB_UTILS (SQLAlchemy): Error preparing update for job (PK ID: {job_pk_id}, Ext. ID: {external_id}): {e}", exc_info=True)
            raise # Re-raise to allow the caller (Airflow task) to handle transaction rollback

    log.info(f"DB_UTILS (SQLAlchemy): Prepared {updated_count} jobs for embedding update in the current session.")
    return updated_count


def upsert_jobs_sqlalchemy(db_session: Session, jobs_to_upsert_data: list[dict]) -> dict:
    """
    Upserts job data (including embeddings) into the database using an existing SQLAlchemy session.
    If a job with the same 'external_job_id' exists, it updates the record.
    Otherwise, it inserts a new record.
    Assumes the 'Job' model may have 'last_updated_ts'.

    Args:
        db_session (Session): The active SQLAlchemy session.
        jobs_to_upsert_data (list[dict]): List of job entries. Expected keys:
            'external_job_id', 'title', 'company', 'location', 
            'job_description_text', 'embedding_vector' (list of floats),
            'last_updated_raw' (string or datetime object for timestamp).

    Returns:
        dict: {'new_jobs_added': count_new, 'jobs_updated': count_updated}.
    """
    if not jobs_to_upsert_data:
        log.info("DB_UTILS (SQLAlchemy - Upsert): No job data provided to upsert.")
        return {'new_jobs_added': 0, 'jobs_updated': 0}

    log.info(f"DB_UTILS (SQLAlchemy - Upsert): Preparing to process {len(jobs_to_upsert_data)} job entries for upsert.")
    
    count_new = 0
    count_updated = 0

    for job_data in jobs_to_upsert_data:
        external_id = job_data.get('external_job_id')
        if not external_id:
            log.warning(f"DB_UTILS (SQLAlchemy - Upsert): Skipping job due to missing 'external_job_id': "
                        f"{job_data.get('title', 'N/A')}")
            continue

        embedding_vector = job_data.get('embedding_vector')
        if embedding_vector is None or not isinstance(embedding_vector, list):
            log.warning(f"DB_UTILS (SQLAlchemy - Upsert): Skipping job {external_id} due to missing or invalid 'embedding_vector'.")
            continue
            
        last_updated_raw = job_data.get('last_updated_raw')
        parsed_last_updated_ts = None
        if isinstance(last_updated_raw, str) and last_updated_raw.strip():
            try:
                parsed_last_updated_ts = pd.to_datetime(last_updated_raw).to_pydatetime()
            except Exception as e_ts:
                log.warning(f"DB_UTILS (SQLAlchemy - Upsert): Could not parse 'last_updated_raw' string '{last_updated_raw}' "
                            f"for job_id {external_id}. Error: {e_ts}. Using current time as fallback.")
                parsed_last_updated_ts = datetime.now()
        elif isinstance(last_updated_raw, datetime):
             parsed_last_updated_ts = last_updated_raw
        else:
            log.debug(f"DB_UTILS (SQLAlchemy - Upsert): 'last_updated_raw' for job_id {external_id} is missing or invalid. Using current time as fallback.")
            parsed_last_updated_ts = datetime.now()

        try:
            existing_job = db_session.query(Job).filter(Job.external_job_id == external_id).first()

            if existing_job:
                log.debug(f"DB_UTILS (SQLAlchemy - Upsert): Updating existing job with external_id: {external_id}")
                existing_job.title = job_data.get('title', existing_job.title)
                existing_job.company = job_data.get('company', existing_job.company)
                existing_job.location = job_data.get('location', existing_job.location)
                existing_job.job_description_text = job_data.get('job_description_text', existing_job.job_description_text)
                existing_job.job_description_embedding = embedding_vector
                
                if hasattr(existing_job, 'last_updated_ts'): # Check if the model attribute exists
                    existing_job.last_updated_ts = parsed_last_updated_ts
                
                count_updated += 1
            else:
                log.debug(f"DB_UTILS (SQLAlchemy - Upsert): Creating new job with external_id: {external_id}")
                job_attributes = {
                    'external_job_id': external_id,
                    'title': job_data.get('title'),
                    'company': job_data.get('company'),
                    'location': job_data.get('location'),
                    'job_description_text': job_data.get('job_description_text'),
                    'job_description_embedding': embedding_vector,
                }
                if hasattr(Job, 'last_updated_ts'): # Check if the class attribute exists
                     job_attributes['last_updated_ts'] = parsed_last_updated_ts

                new_job = Job(**job_attributes)
                db_session.add(new_job)
                count_new += 1
        except Exception as e:
            log.error(f"DB_UTILS (SQLAlchemy - Upsert): Error processing job with external_id {external_id} in session: {e}", exc_info=True)
            raise 

    log.info(f"DB_UTILS (SQLAlchemy - Upsert): Finished preparing {count_new} new and {count_updated} updated job entries in session.")
    return {'new_jobs_added': count_new, 'jobs_updated': count_updated}


