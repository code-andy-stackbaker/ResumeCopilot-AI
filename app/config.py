import os
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration variables
MODEL_NAME = os.getenv("MODEL_NAME", None)  # No default in production!
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", None)
METADATA_PATH = os.getenv("METADATA_PATH", None)
Recommender_MODEL_NAME = os.getenv("MODEL_NAME", None)  # No default in production!


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