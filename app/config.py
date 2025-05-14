import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()


# Define the project root dynamically and robustly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Assuming app is one level below
print ("the project root:", PROJECT_ROOT)

def root_path(*args):
    return os.path.join(PROJECT_ROOT, *args)

# Default relative paths (relative to project root)
DEFAULT_FAISS_INDEX_PATH = root_path("app", "recommender", "model", "job_index.faiss")
DEFAULT_METADATA_PATH = root_path("app", "recommender", "model", "job_metadata.csv")



# Configuration variables
MODEL_NAME = os.getenv("MODEL_NAME", None)  # No default in production!
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", DEFAULT_FAISS_INDEX_PATH)
METADATA_PATH = os.getenv("METADATA_PATH", DEFAULT_METADATA_PATH)
Recommender_MODEL_NAME = os.getenv("Recommender_MODEL_NAME", "all-MiniLM-L6-v2" )  # No default in production!

# --- Database Configuration ---
DATABASE_USER = os.getenv("DB_USER", "job_user") # Default values can be provided
DATABASE_PASSWORD = os.getenv("DB_PASSWORD", "job_password")
DATABASE_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_PORT = os.getenv("DB_PORT", "5432")
DATABASE_NAME = os.getenv("DB_NAME", "job_listings_db")


DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# --- Embedding Configuration ---
# Embedding dimension for the recommender system's sentence transformer model.
# For "all-MiniLM-L6-v2", this is 384.
# For OpenAI "text-embedding-ada-002", it's 1536.
EMBEDDING_DIMENSION_RECOMMENDER = int(os.getenv("EMBEDDING_DIMENSION_RECOMMENDER", 384))


# Validation (Crucial!)
if not MODEL_NAME:
  logging.warning("MODEL_NAME not set in environment. Application may not function correctly.")
else:
  logging.info(f"Using MODEL_NAME: {MODEL_NAME}")

if not Recommender_MODEL_NAME:
  logging.warning("MODEL_NAME not set in environment. Application may not function correctly.")
else:
  logging.info(f"Using MODEL_NAME: { Recommender_MODEL_NAME}")    

if not FAISS_INDEX_PATH:
  logging.warning("FAISS_INDEX_PATH not set in environment. Application may not function correctly.")
else:
  logging.info(f"Using FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")

if not METADATA_PATH:
  logging.warning("METADATA_PATH not set in environment. Application may not function correctly.")
else:
  logging.info(f"Using METADATA_PATH: {METADATA_PATH}")