# app/config.py
import os
from dotenv import load_dotenv
import logging
import sys # For exiting on critical config error

# It's generally better to configure logging once at your application's entry point (e.g., main.py or your script's main block).
# However, if this config is imported by scripts that might not have logging set up, this can be a fallback.
# For now, I'll keep it as you have it.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Good practice to use __name__ for the logger

# --- Define Project Root and Load .env ---
# Your existing robust way to define PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"[DEBUG config.py] PROJECT_ROOT determined as: {PROJECT_ROOT}")

# Construct the explicit path to the .env file located in the project root
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
print(f"[DEBUG config.py] Attempting to load .env from: {dotenv_path}")

if os.path.exists(dotenv_path):
    print(f"[DEBUG config.py] .env file found at: {dotenv_path}")
    # override=True ensures that variables from .env take precedence over existing environment vars.
    # verbose=True will print a message if .env is loaded.
    loaded_successfully = load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
    print(f"[DEBUG config.py] load_dotenv execution result (True if .env was loaded): {loaded_successfully}")
else:
    print(f"[DEBUG config.py] .env file NOT found at: {dotenv_path}. Configuration will rely on pre-existing environment variables.")
    # No error here yet, as env vars could be set externally (e.g., in Docker, CI/CD)

# --- Retrieve Configuration Variables (STRICTLY from environment/.env) ---
# For essential variables, we will not provide default fallbacks here.
# They MUST be in the .env file or the system environment.

DATABASE_USER = os.getenv("DB_USER")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_HOST = os.getenv("DB_HOST")
DATABASE_PORT = os.getenv("DB_PORT")
DATABASE_NAME = os.getenv("DB_NAME")

Recommender_MODEL_NAME = os.getenv("Recommender_MODEL_NAME")
# MODEL_NAME from your original script, keep if used distinctly
MODEL_NAME = os.getenv("MODEL_NAME")
# FAISS_INDEX_PATH might become less relevant as we fully transition to DB
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")

# METADATA_PATH is crucial for the ingestion script
METADATA_PATH = os.getenv("METADATA_PATH")
print(f"[DEBUG config.py] Value of METADATA_PATH from os.getenv: {METADATA_PATH}")

# EMBEDDING_DIMENSION_RECOMMENDER also needs to be in .env
EMBEDDING_DIMENSION_RECOMMENDER_STR = os.getenv("EMBEDDING_DIMENSION_RECOMMENDER")

# --- Validate Essential Configurations ---
# List all configurations that your application absolutely needs to start.
essential_configs = {
    "DB_USER": DATABASE_USER,
    "DB_PASSWORD": DATABASE_PASSWORD,
    "DB_HOST": DATABASE_HOST,
    "DB_PORT": DATABASE_PORT,
    "DB_NAME": DATABASE_NAME,
    "Recommender_MODEL_NAME": Recommender_MODEL_NAME,
    "METADATA_PATH": METADATA_PATH,
    "EMBEDDING_DIMENSION_RECOMMENDER": EMBEDDING_DIMENSION_RECOMMENDER_STR
    # Add other essential variables here, e.g., MODEL_NAME if it's always required
}

missing_configs = [key for key, value in essential_configs.items() if value is None]

if missing_configs:
    error_message = (
        f"CRITICAL CONFIGURATION ERROR: The following essential environment variables are not set "
        f"or not loaded from .env: {', '.join(missing_configs)}. "
        f"Please define them in your .env file (expected at: {dotenv_path}) or system environment."
    )
    logger.critical(error_message)
    # For scripts or critical startup, raising an error is appropriate.
    raise ValueError(error_message)

# --- Process and Construct Derived Configurations (if all essential configs are present) ---
try:
    EMBEDDING_DIMENSION_RECOMMENDER = int(EMBEDDING_DIMENSION_RECOMMENDER_STR)
except ValueError:
    error_message = (
        f"CRITICAL CONFIGURATION ERROR: EMBEDDING_DIMENSION_RECOMMENDER "
        f"('{EMBEDDING_DIMENSION_RECOMMENDER_STR}') is not a valid integer. "
        f"Please set it correctly in your .env file."
    )
    logger.critical(error_message)
    raise ValueError(error_message)

DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# --- Log Loaded Configurations (Your original logging, adapted for clarity) ---
logger.info("--- Successfully Loaded Configurations ---")
if MODEL_NAME: # If MODEL_NAME is optional, this fine. If required, add to essential_configs.
    logger.info(f"Using MODEL_NAME: {MODEL_NAME}")
else:
    logger.warning("MODEL_NAME not set in environment (this may be optional).")

logger.info(f"Using Recommender_MODEL_NAME: {Recommender_MODEL_NAME}")

if FAISS_INDEX_PATH: # If FAISS_INDEX_PATH is optional, this fine.
    logger.info(f"Using FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
else:
    logger.warning("FAISS_INDEX_PATH not set (this may be optional or becoming obsolete).")

logger.info(f"Using METADATA_PATH: {METADATA_PATH}") # Should definitely be set now
logger.info(f"Using EMBEDDING_DIMENSION_RECOMMENDER: {EMBEDDING_DIMENSION_RECOMMENDER}")
logger.info(f"Database URL configured for host: {DATABASE_HOST}")
logger.info("--- End of Configurations ---")

# Your root_path helper function (useful for other parts of your app potentially)
def root_path(*args):
    return os.path.join(PROJECT_ROOT, *args)

# Example of using it if needed elsewhere, though FAISS/metadata paths are now directly from .env
# DEFAULT_FAISS_INDEX_PATH = root_path("app", "recommender", "model", "job_index.faiss")
# DEFAULT_METADATA_PATH = root_path("app", "recommender", "model", "job_metadata.csv")