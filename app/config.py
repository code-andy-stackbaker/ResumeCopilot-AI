# LLM_DeepLearning/app/config.py
import os
# import logging # Comment out logging for this debug version
import sys

# --- Start of app/config.py parsing ---
print("--- [APP_CONFIG_PRINT_DEBUG] app/config.py parsing started ---", flush=True)

# --- Define Project Root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"--- [APP_CONFIG_PRINT_DEBUG] PROJECT_ROOT determined as: {PROJECT_ROOT}", flush=True)

# --- DOTENV LOADING SECTION - REMAINS TEMPORARILY DISABLED FOR TESTING ---
print("--- [APP_CONFIG_PRINT_DEBUG] python-dotenv load_dotenv() calls are currently commented out.", flush=True)
# ... (load_dotenv lines remain commented) ...

print("--- [APP_CONFIG_PRINT_DEBUG] Attempting to retrieve variables using os.getenv() ---", flush=True)

DATABASE_USER = os.getenv("DB_USER")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('DB_USER'): '{DATABASE_USER}' (Type: {type(DATABASE_USER)})", flush=True)

DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('DB_PASSWORD'): '{DATABASE_PASSWORD}' (Type: {type(DATABASE_PASSWORD)})", flush=True)

DATABASE_HOST = os.getenv("DB_HOST")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('DB_HOST'): '{DATABASE_HOST}' (Type: {type(DATABASE_HOST)})", flush=True)

DATABASE_PORT = os.getenv("DB_PORT")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('DB_PORT'): '{DATABASE_PORT}' (Type: {type(DATABASE_PORT)})", flush=True)

DATABASE_NAME = os.getenv("DB_NAME")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('DB_NAME'): '{DATABASE_NAME}' (Type: {type(DATABASE_NAME)})", flush=True)

Recommender_MODEL_NAME = os.getenv("Recommender_MODEL_NAME")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('Recommender_MODEL_NAME'): '{Recommender_MODEL_NAME}' (Type: {type(Recommender_MODEL_NAME)})", flush=True)

METADATA_PATH = os.getenv("METADATA_PATH")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('METADATA_PATH'): '{METADATA_PATH}' (Type: {type(METADATA_PATH)})", flush=True)

EMBEDDING_DIMENSION_RECOMMENDER_STR = os.getenv("EMBEDDING_DIMENSION_RECOMMENDER")
print(f"--- [APP_CONFIG_PRINT_DEBUG] Value of os.getenv('EMBEDDING_DIMENSION_RECOMMENDER'): '{EMBEDDING_DIMENSION_RECOMMENDER_STR}' (Type: {type(EMBEDDING_DIMENSION_RECOMMENDER_STR)})", flush=True)

print("--- [APP_CONFIG_PRINT_DEBUG] Finished retrieving variables. Now proceeding to validation. ---", flush=True)

# --- Validate Essential Configurations ---
essential_configs = {
    "DB_USER": DATABASE_USER,
    "DB_PASSWORD": DATABASE_PASSWORD,
    "DB_HOST": DATABASE_HOST,
    "DB_PORT": DATABASE_PORT,
    "DB_NAME": DATABASE_NAME,
    "Recommender_MODEL_NAME": Recommender_MODEL_NAME,
    "METADATA_PATH": METADATA_PATH,
    "EMBEDDING_DIMENSION_RECOMMENDER": EMBEDDING_DIMENSION_RECOMMENDER_STR
}
missing_configs = [key for key, value in essential_configs.items() if value is None]

if missing_configs:
    PROJECT_ROOT_FOR_ERROR_MSG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dotenv_path_for_error_msg = os.path.join(PROJECT_ROOT_FOR_ERROR_MSG, '.env')
    error_message = (
        f"CRITICAL CONFIGURATION ERROR (print debug): The following essential environment variables are reported as None by os.getenv(): "
        f"{', '.join(missing_configs)}. "
        f"These should be available in the system environment (e.g., via Docker Compose env_file from LLM_DeepLearning/.env). "
        f"The config.py script's own attempt to load from '{dotenv_path_for_error_msg}' is currently disabled or did not find the file."
    )
    print(f"--- [APP_CONFIG_PRINT_DEBUG] ERROR: {error_message}", flush=True) # Print the error too
    raise ValueError(error_message)

# --- Process and Construct Derived Configurations (if all essential configs are present) ---
print("--- [APP_CONFIG_PRINT_DEBUG] Successfully passed essential config validation. Processing derived configurations. ---", flush=True)
try:
    EMBEDDING_DIMENSION_RECOMMENDER = int(EMBEDDING_DIMENSION_RECOMMENDER_STR)
except ValueError:
    error_message = (
        f"CRITICAL CONFIGURATION ERROR: EMBEDDING_DIMENSION_RECOMMENDER "
        f"('{EMBEDDING_DIMENSION_RECOMMENDER_STR}') is not a valid integer."
    )
    print(f"--- [APP_CONFIG_PRINT_DEBUG] ERROR: {error_message}", flush=True)
    raise ValueError(error_message)

DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

# --- Log Loaded Configurations ---
print(f"--- [APP_CONFIG_PRINT_DEBUG] Database URL configured for host: {DATABASE_HOST}", flush=True)
print("--- [APP_CONFIG_PRINT_DEBUG] End of app/config.py parsing ---", flush=True)

def root_path(*args):
    return os.path.join(PROJECT_ROOT, *args)