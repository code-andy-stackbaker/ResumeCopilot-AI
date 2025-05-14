# scripts/ingest_data.py
import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Add project root to Python path to allow importing app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.session import SessionLocal, engine, Base
from app.database.models import Job # Your Job SQLAlchemy model
from app.config import (
    Recommender_MODEL_NAME,
    EMBEDDING_DIMENSION_RECOMMENDER,
    # We need to define where your CSV data is.
    # Let's assume you have a CSV_FILE_PATH in your config or define it here.
    # For now, I'll use the METADATA_PATH if it points to your jobs CSV.
    # Ensure this path is correct for your actual jobs data CSV.
    METADATA_PATH as CSV_FILE_PATH
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the table schema exists (Alembic should have created it)
# For safety, you can call Base.metadata.create_all(bind=engine)
# but it's better if Alembic is the sole manager of schema.
# We'll assume Alembic has run.

def get_embedding_model(model_name: str):
    """Loads and returns the SentenceTransformer model."""
    try:
        logger.info(f"Loading sentence transformer model: {model_name}...")
        model = SentenceTransformer(model_name)
        logger.info("Sentence transformer model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading sentence transformer model: {e}", exc_info=True)
        raise

def ingest_data(db_session, model, csv_path: str):
    """
    Reads data from a CSV file, generates embeddings, and ingests into the database.
    """
    try:
        logger.info(f"Reading job data from CSV: {csv_path}")
        if not os.path.exists(csv_path):
          logger.error(f"CSV file not found at {csv_path}. Please check the path.")
          return 0

        df = pd.read_csv(csv_path)
        logger.info(f"Found {len(df)} records in CSV file.")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}", exc_info=True)
        return 0

    jobs_ingested_count = 0
    for index, row in df.iterrows():
        try:
            # --- Adapt these column names to match your CSV file ---
            title = row.get('title', 'N/A')
            company = row.get('company', None)
            location = row.get('location', None)
            # IMPORTANT: Ensure 'job_description_text' (or your equivalent column) exists and is not empty.
            job_desc_text = row.get('job_description_text') # Or 'description', 'job_desciption' etc.
            external_id = str(row.get('external_job_id', f"csv_job_{index}")) # Example: use index if no external_id

            if not job_desc_text or pd.isna(job_desc_text):
                logger.warning(f"Skipping row {index+2} due to missing job description text. Title: {title}")
                continue

            # Generate embedding for the job description
            # The model.encode() method returns a list of embeddings if given a list of texts.
            # For a single text, it returns a single embedding (numpy array).
            embedding_array = model.encode(str(job_desc_text))

            # Ensure embedding is a list of floats/ints as expected by pgvector from numpy array
            # pgvector Python client handles numpy arrays directly.

            # Check if job with this external_id already exists to avoid duplicates
            existing_job = db_session.query(Job).filter(Job.external_job_id == external_id).first()
            if existing_job:
                logger.info(f"Job with external_id '{external_id}' (Title: {title}) already exists. Skipping.")
                continue

            new_job = Job(
                external_job_id=external_id,
                title=str(title),
                company=str(company) if company else None,
                location=str(location) if location else None,
                job_description_text=str(job_desc_text),
                job_description_embedding=embedding_array # Pass the numpy array directly
            )
            db_session.add(new_job)
            jobs_ingested_count += 1

            if (index + 1) % 100 == 0: # Log progress every 100 records
                logger.info(f"Processed {index + 1}/{len(df)} records. Added {jobs_ingested_count} new jobs so far.")

        except Exception as e:
            logger.error(f"Error processing row {index+2} (Title: {title}): {e}", exc_info=True)
            # Decide if you want to rollback or continue. For bulk ingestion, often we log and continue.
            # db_session.rollback() # Could rollback this specific item or a batch

    try:
        if jobs_ingested_count > 0:
            logger.info(f"Committing {jobs_ingested_count} new jobs to the database...")
            db_session.commit()
            logger.info("Data committed successfully.")
        else:
            logger.info("No new jobs to commit.")
    except Exception as e:
        logger.error(f"Error committing data to database: {e}", exc_info=True)
        db_session.rollback()
        logger.info("Database transaction rolled back.")
        return 0

    return jobs_ingested_count

if __name__ == "__main__":
    logger.info("Starting data ingestion process...")

    # Ensure CSV_FILE_PATH is correctly pointing to your jobs data.
    # This might be app/recommender/model/job_metadata.csv or app/langchain_qa/data/jobs.csv
    # depending on which one contains the full job details including descriptions.
    # You might need to consolidate your job data first if it's fragmented.

    # For this example, let's assume CSV_FILE_PATH is correctly set in config
    # or you can hardcode it here for the script's purpose if it's fixed:
    # For example:
    # resolved_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'recommender', 'model', 'job_metadata.csv'))

    # Let's try to make CSV_FILE_PATH relative to the project root if it's a relative path in config
    if not os.path.isabs(CSV_FILE_PATH):
         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
         resolved_csv_path = os.path.join(project_root, CSV_FILE_PATH)
    else:
        resolved_csv_path = CSV_FILE_PATH

    if not os.path.exists(resolved_csv_path):
        logger.error(f"CRITICAL: CSV file for ingestion not found at '{resolved_csv_path}'.")
        logger.error("Please ensure the METADATA_PATH in your .env or app/config.py points to the correct CSV file containing job descriptions.")
        sys.exit(1)

    embedding_model = None
    db = None
    try:
        embedding_model = get_embedding_model(Recommender_MODEL_NAME)
        db = SessionLocal() # Get a database session

        # Optional: Create tables if they don't exist (Alembic should handle this in prod)
        # Base.metadata.create_all(bind=engine) 
        # logger.info("Ensured tables exist (SQLAlchemy create_all).")

        num_ingested = ingest_data(db, embedding_model, resolved_csv_path)
        logger.info(f"Ingestion process completed. {num_ingested} new jobs were added to the database.")

    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)
    finally:
        if db:
            db.close()
        logger.info("Database session closed.")