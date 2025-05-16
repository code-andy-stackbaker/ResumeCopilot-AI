# File: PROJECT_ROOT/utils/embedding_utils.py
import logging
import os
import sys
from sentence_transformers import SentenceTransformer
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
        print(f"ERROR in embedding_utils.py: Could not import from app.config. "
              f"Ensure PROJECT_ROOT is in sys.path. Attempted fallback path: {_PROJECT_ROOT_FOR_UTIL}. Error: {e_inner}")
        raise

log = logging.getLogger(__name__) # Uses the logger configured in app/config.py

# In-memory cache for the model to avoid reloading it multiple times within the same process
_model_cache = {}

def get_embedding_model(model_name: str = None) -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer model using the name from app.config
    or an optionally provided model_name. Caches the model in memory.
    """
    # Use provided model_name or fallback to the one from config
    effective_model_name = model_name if model_name else Recommender_MODEL_NAME
    
    if effective_model_name in _model_cache:
        log.info(f"EMBEDDING_UTILS: Returning cached SentenceTransformer model: '{effective_model_name}'.")
        return _model_cache[effective_model_name]
    
    try:
        log.info(f"EMBEDDING_UTILS: Loading SentenceTransformer model: '{effective_model_name}'...")
        model = SentenceTransformer(effective_model_name)
        log.info(f"EMBEDDING_UTILS: SentenceTransformer model '{effective_model_name}' loaded successfully.")
        
        # Validate the loaded model's embedding dimension against the configured dimension
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
        log.warning("EMBEDDING_UTILS: Invalid or empty text provided for embedding. Returning None or empty array.")
        
    embedding_array = model.encode(str(text_to_embed)) # Returns a numpy array for a single string
    log.debug(f"EMBEDDING_UTILS: Generated embedding for text snippet: '{str(text_to_embed)[:50]}...'")
    return embedding_array

# Optional: If you often process batches, a batch function is more efficient
def generate_batch_embeddings(texts: list[str], model: SentenceTransformer) -> list[np.ndarray]:
    """
    Generates embeddings for a list of texts using the provided pre-loaded model.
    Returns a list of NumPy arrays.
    """
    if not texts:
        log.info("EMBEDDING_UTILS: No texts provided for batch embedding generation.")
        return []
    
    log.info(f"EMBEDDING_UTILS: Generating batch embeddings for {len(texts)} texts...")
    embedding_arrays = model.encode([str(t) for t in texts], show_progress_bar=False) # List of numpy arrays
    log.info(f"EMBEDDING_UTILS: Batch embeddings generated for {len(texts)} texts.")
    return embedding_arrays