# File: PROJECT_ROOT/utils/embedding_utils.py
import logging
import os
import sys
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np

# Attempt to import config. If this script is run directly, it might need help finding PROJECT_ROOT.
# The primary caller (DAG or another script) should ensure PROJECT_ROOT is in sys.path.
try:
    from app.config import Recommender_MODEL_NAME, EMBEDDING_DIMENSION_RECOMMENDER
except ImportError:
    # Fallback for trying to run/test this util standalone (not recommended for DAGs)
    # Assumes 'utils' is one level down from project_root.
    _PROJECT_ROOT_FOR_UTIL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT_FOR_UTIL not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_FOR_UTIL)
    try:
        from app.config import Recommender_MODEL_NAME, EMBEDDING_DIMENSION_RECOMMENDER
    except ImportError as e_inner:
        # Using print here as logger might not be configured if app.config fails to load
        print(f"ERROR in embedding_utils.py: Could not import from app.config. "
              f"Ensure PROJECT_ROOT is in sys.path. Attempted fallback path: {_PROJECT_ROOT_FOR_UTIL}. Error: {e_inner}")
        raise

log = logging.getLogger(__name__) # Uses the logger configured in app/config.py or basicConfig if standalone

# In-memory cache for the model to avoid reloading it multiple times within the same process
_model_cache = {}

def get_embedding_model(model_name: str = None) -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer model using the name from app.config
    or an optionally provided model_name. Caches the model in memory.
    """
    effective_model_name = model_name if model_name else Recommender_MODEL_NAME
    
    if effective_model_name in _model_cache:
        log.info(f"EMBEDDING_UTILS: Returning cached SentenceTransformer model: '{effective_model_name}'.")
        return _model_cache[effective_model_name]
    
    try:
        log.info(f"EMBEDDING_UTILS: Loading SentenceTransformer model: '{effective_model_name}'...")
        model = SentenceTransformer(effective_model_name)
        log.info(f"EMBEDDING_UTILS: SentenceTransformer model '{effective_model_name}' loaded successfully.")
        
        model_output_dim = model.get_sentence_embedding_dimension()
        if model_output_dim != EMBEDDING_DIMENSION_RECOMMENDER:
            error_msg = (f"EMBEDDING_UTILS: Model '{effective_model_name}' output dimension ({model_output_dim}) "
                         f"does not match configured EMBEDDING_DIMENSION_RECOMMENDER ({EMBEDDING_DIMENSION_RECOMMENDER}). "
                         f"Please check your model or .env configuration.")
            log.error(error_msg)
            raise ValueError(error_msg)
            
        _model_cache[effective_model_name] = model
        return model
    except Exception as e:
        log.error(f"EMBEDDING_UTILS: Error loading SentenceTransformer model '{effective_model_name}': {e}", exc_info=True)
        raise

def generate_text_embedding(text_to_embed: str, model: SentenceTransformer) -> np.ndarray:
    """
    Generates an embedding for a single piece of text using the provided pre-loaded model.
    Returns a NumPy array.
    """
    if not text_to_embed or not isinstance(text_to_embed, str):
        log.warning("EMBEDDING_UTILS: Invalid or empty text provided for single text embedding. Encoding as is.")
        # Or: return np.zeros(EMBEDDING_DIMENSION_RECOMMENDER) if a zero vector is preferred for invalid input
        
    embedding_array = model.encode(str(text_to_embed)) 
    log.debug(f"EMBEDDING_UTILS: Generated single embedding for text snippet: '{str(text_to_embed)[:50]}...'")
    return embedding_array

def generate_batch_embeddings(texts: list[str], model: SentenceTransformer) -> list[np.ndarray]:
    """
    Generates embeddings for a list of texts using the provided pre-loaded model.
    Returns a list of NumPy arrays.
    """
    if not texts:
        log.info("EMBEDDING_UTILS: No texts provided for batch embedding generation.")
        return []
    
    log.info(f"EMBEDDING_UTILS: Generating batch embeddings for {len(texts)} texts...")
    # Ensure all inputs are strings for the model
    str_texts = [str(t) if t is not None else "" for t in texts]
    embedding_arrays = model.encode(str_texts, show_progress_bar=False) # List of numpy arrays
    log.info(f"EMBEDDING_UTILS: Batch embeddings generated for {len(texts)} texts.")
    return embedding_arrays

# --- NEW FUNCTION ---
def generate_embeddings_for_texts(texts: list[str], model_name: str = None) -> list[list[float]]:
    """
    Generates embeddings for a list of texts, handling model loading and caching.
    Returns a list of embedding vectors (each as a list of floats).
    """
    if not texts:
        log.info("EMBEDDING_UTILS (generate_embeddings_for_texts): No texts provided. Returning empty list.")
        return []
    
    # Get the model (uses cache, configured name by default, or specified model_name)
    try:
        model = get_embedding_model(model_name=model_name)
    except Exception as e: # Catch errors from model loading specifically
        log.error(f"EMBEDDING_UTILS (generate_embeddings_for_texts): Failed to get embedding model. Error: {e}", exc_info=True)
        raise # Re-raise to fail the calling process (e.g., Airflow task)

    log.info(f"EMBEDDING_UTILS (generate_embeddings_for_texts): Model '{model_name or Recommender_MODEL_NAME}' retrieved. "
             f"Proceeding to generate embeddings for {len(texts)} texts.")
    
    # Generate embeddings as numpy arrays
    # generate_batch_embeddings expects a list of strings and a model instance
    embedding_arrays = generate_batch_embeddings(texts, model) # texts are already asserted to be list[str] by type hint
    
    if len(embedding_arrays) != len(texts):
        # This case should ideally be handled by generate_batch_embeddings or model.encode
        # but an extra check doesn't hurt.
        log.warning(f"EMBEDDING_UTILS (generate_embeddings_for_texts): Mismatch in expected "
                    f"({len(texts)}) and generated ({len(embedding_arrays)}) embeddings count.")
        # Decide on handling: raise error or try to return partial (current: proceed with what was returned)

    # Convert list[np.ndarray] to list[list[float]] for pgvector/JSON compatibility
    embeddings_as_lists = [arr.tolist() for arr in embedding_arrays]
    
    log.info(f"EMBEDDING_UTILS (generate_embeddings_for_texts): Successfully generated and converted {len(embeddings_as_lists)} embeddings into list[list[float]].")
    return embeddings_as_lists