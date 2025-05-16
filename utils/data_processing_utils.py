# File: PROJECT_ROOT/utils/data_processing_utils.py
import pandas as pd
import logging
import os
import sys
from datetime import datetime

# Attempt to import config. The primary caller (DAG or another script)
# should ensure PROJECT_ROOT is in sys.path.
try:
    # We need METADATA_PATH for the default CSV location and PROJECT_ROOT to resolve it if relative.
    from app.config import METADATA_PATH as DEFAULT_CSV_PATH, PROJECT_ROOT
except ImportError:
    # Fallback for trying to run/test this util standalone
    _PROJECT_ROOT_FOR_UTIL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT_FOR_UTIL not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_FOR_UTIL)
    try:
        from app.config import METADATA_PATH as DEFAULT_CSV_PATH, PROJECT_ROOT
    except ImportError as e_inner:
        # Use a basic logger if app.config logger isn't available yet
        _temp_log = logging.getLogger(__name__)
        _temp_log.error(f"ERROR in data_processing_utils.py: Could not import from app.config. "
                        f"Ensure PROJECT_ROOT is in sys.path. Attempted fallback path: {_PROJECT_ROOT_FOR_UTIL}. Error: {e_inner}")
        raise

# Assuming app.config has initialized logging.basicConfig
log = logging.getLogger(__name__)

def fetch_and_prepare_job_data_from_csv(csv_file_path: str = None) -> list[dict]:
    resolved_csv_path = csv_file_path
    if resolved_csv_path is None:
        if not DEFAULT_CSV_PATH: # Check if DEFAULT_CSV_PATH itself is None or empty from config
            log.error("DATA_PROCESSING_UTILS: DEFAULT_CSV_PATH (METADATA_PATH in config) is not set.")
            return []
        if not os.path.isabs(DEFAULT_CSV_PATH):
            if not PROJECT_ROOT: # Check if PROJECT_ROOT is available from config
                 log.error("DATA_PROCESSING_UTILS: PROJECT_ROOT from config is not set, cannot resolve relative DEFAULT_CSV_PATH.")
                 return []
            resolved_csv_path = os.path.join(PROJECT_ROOT, DEFAULT_CSV_PATH)
            log.info(f"DATA_PROCESSING_UTILS: Resolved relative DEFAULT_CSV_PATH to: {resolved_csv_path}")
        else:
            resolved_csv_path = DEFAULT_CSV_PATH
            log.info(f"DATA_PROCESSING_UTILS: Using absolute DEFAULT_CSV_PATH: {resolved_csv_path}")
    elif not os.path.isabs(resolved_csv_path):
        if not PROJECT_ROOT:
            log.error("DATA_PROCESSING_UTILS: PROJECT_ROOT from config is not set, cannot resolve provided relative csv_file_path.")
            return []
        resolved_csv_path = os.path.join(PROJECT_ROOT, resolved_csv_path)
        log.info(f"DATA_PROCESSING_UTILS: Resolved provided relative csv_file_path to: {resolved_csv_path}")
    else:
        log.info(f"DATA_PROCESSING_UTILS: Using provided absolute csv_file_path: {resolved_csv_path}")

    log.info(f"DATA_PROCESSING_UTILS: Attempting to read job data from CSV: {resolved_csv_path}")
    if not os.path.exists(resolved_csv_path):
        log.error(f"DATA_PROCESSING_UTILS: CSV file not found at {resolved_csv_path}. Please check the path.")
        return []

    try:
        df = pd.read_csv(resolved_csv_path)
        log.info(f"DATA_PROCESSING_UTILS: Found {len(df)} records in CSV file: {resolved_csv_path}")
    except Exception as e:
        log.error(f"DATA_PROCESSING_UTILS: Error reading CSV file at {resolved_csv_path}: {e}", exc_info=True)
        return []

    jobs_data_prepared = []
    for index, row in df.iterrows():
        try:
            # --- Column Name Handling (adapting from your ingest_data.py) ---
            title = str(row.get('title', 'N/A')) # Default to 'N/A' if missing, ensure string
            company = str(row.get('company', '')) if pd.notna(row.get('company')) else None
            location = str(row.get('location', '')) if pd.notna(row.get('location')) else None
            
            # Prioritize 'job_description_text' then 'description'
            job_desc_text_raw = row.get('job_description_text', row.get('description'))
            
            # Prioritize 'external_job_id' then 'job_id', then generate a fallback
            external_id_raw = row.get('external_job_id', row.get('job_id'))
            if pd.isna(external_id_raw) or str(external_id_raw).strip() == "":
                external_id = f"csv_job_index_{index}" # Fallback if primary IDs are missing
                log.debug(f"DATA_PROCESSING_UTILS: Using fallback ID '{external_id}' for row {index+2}.")
            else:
                external_id = str(external_id_raw).strip()

            if pd.isna(job_desc_text_raw) or str(job_desc_text_raw).strip() == "":
                log.warning(f"DATA_PROCESSING_UTILS: Skipping row {index+2} (External ID: {external_id}, Title: {title}) due to missing or empty job description text.")
                continue

            job_desc_text = str(job_desc_text_raw) # Ensure it's a string

            # Prepare structured dictionary
            job_entry = {
                'external_job_id': external_id,
                'title': title,
                'company': company,
                'location': location,
                'job_description_text': job_desc_text,
                'last_updated_raw': row.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')) # Store raw, parse later
            }
            jobs_data_prepared.append(job_entry)

            if (index + 1) % 200 == 0: # Log progress less frequently for a util
                log.info(f"DATA_PROCESSING_UTILS: Prepared {index + 1}/{len(df)} job entries from CSV.")

        except Exception as e:
            # Log error for the specific row and continue if possible
            log.error(f"DATA_PROCESSING_UTILS: Error processing CSV row {index+2} (Title: {title if 'title' in locals() else 'N/A'}): {e}", exc_info=True)
            continue # Skip to the next row

    log.info(f"DATA_PROCESSING_UTILS: Successfully prepared {len(jobs_data_prepared)} job entries from {len(df)} CSV records.")
    return jobs_data_prepared


