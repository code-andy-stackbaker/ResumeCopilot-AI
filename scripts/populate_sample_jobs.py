# File: PROJECT_ROOT/scripts/populate_sample_jobs.py

import logging
import os
import sys
from datetime import datetime
from uuid import uuid4

from faker import Faker # For generating fake data

# --- Add project root to sys.path to find custom modules ---
# This script is expected to be in PROJECT_ROOT/scripts/
# So, PROJECT_ROOT is one level up from os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

try:
    # Import SQLAlchemy SessionLocal for creating DB sessions
    from app.database.session import SessionLocal
    # Import your Job model
    from app.database.models import Job
except ImportError as e:
    log.error(f"SCRIPT IMPORT ERROR: Critical modules not found. "
              f"Ensure PROJECT_ROOT ('{PROJECT_ROOT}') is correct and contains 'app/database/session.py' "
              f"and 'app/database/models.py'. Error: {e}")
    log.error(f"Current sys.path for script: {sys.path}")
    sys.exit(1) # Exit if essential imports fail

fake = Faker()

def create_fake_job_data() -> dict:
    """Generates a dictionary with fake job data."""
    title = fake.job()
    company = fake.company()
    return {
        'external_job_id': f"EXT-{uuid4().hex[:10].upper()}", # Unique external ID
        'title': title,
        'company': company,
        'location': fake.city() + ", " + fake.state_abbr(),
        'job_description_text': f"About the role: {title} at {company}.\n"
                                f"{fake.bs()}.\n"
                                f"Responsibilities include: {fake.sentence(nb_words=10)}.\n"
                                f"Requirements: {fake.sentence(nb_words=8)}.\n"
                                f"We offer: {fake.sentence(nb_words=7)}.",
        'job_description_embedding': None, # Crucially, set to None
        'last_updated_ts': datetime.now(),
        # Add other fields as per your Job model, e.g.:
        # 'salary_min': fake.random_int(min=40000, max=80000, step=1000),
        # 'salary_max': fake.random_int(min=85000, max=150000, step=1000),
        # 'currency': 'USD',
        # 'job_type': fake.random_element(elements=('Full-time', 'Contract', 'Part-time')),
        # 'url': fake.url(),
    }

def populate_jobs(db_session: SessionLocal, num_jobs: int = 50):
    """Populates the database with a specified number of job records."""
    log.info(f"Starting to populate database with {num_jobs} job records...")
    jobs_added_count = 0
    for i in range(num_jobs):
        job_data = create_fake_job_data()
        
        # Basic check to ensure essential text fields are not empty (Faker usually provides this)
        if not job_data['job_description_text'] or not job_data['title']:
            log.warning(f"Skipping job due to empty essential text fields generated by Faker: {job_data.get('external_job_id')}")
            continue

        job_instance = Job(**job_data)
        db_session.add(job_instance)
        jobs_added_count +=1
        if (i + 1) % 10 == 0: # Log progress every 10 jobs
            log.info(f"Prepared {jobs_added_count} jobs for insertion...")

    try:
        db_session.commit()
        log.info(f"Successfully committed {jobs_added_count} new job records to the database.")
    except Exception as e:
        log.error(f"Error during database commit: {e}", exc_info=True)
        db_session.rollback()
        log.info("Database transaction rolled back.")
    finally:
        db_session.close()
        log.info("Database session closed.")

if __name__ == "__main__":
    log.info("Script to populate sample jobs started.")
    # Create a new database session
    db = SessionLocal()
    if db:
        populate_jobs(db_session=db, num_jobs=50)
    else:
        log.error("Failed to create database session. Exiting.")
    log.info("Script finished.")