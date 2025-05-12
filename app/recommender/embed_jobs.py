import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import logging
import os
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default values (can be overridden by environment variables)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32

def create_embeddings_and_index(
    data_path: str,
    output_index_path: str,
    model_name: str = os.getenv("Recommender_MODEL_NAME", DEFAULT_MODEL_NAME),  # From env or default
    batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))  # From env or default
):
    """
    Creates embeddings for job descriptions and builds a FAISS index.

    Args:
        data_path (str): Path to the CSV file containing job descriptions.
        output_index_path (str): Path to save the FAISS index.
        model_name (str, optional): SentenceTransformer model name.
        batch_size (int, optional): Batch size for embedding generation.
    """
    try:
        logging.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        job_descriptions = df["job_desciption"].tolist()

        # Determine device for SentenceTransformer
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
          device = torch.device("mps")
          logging.info("MPS device is available. Using MPS for SentenceTransformer.")
        else:
          device = torch.device("cpu")
          logging.info("MPS device not available. Using CPU for SentenceTransformer.")
          if not torch.backends.mps.is_built():
            logging.warning("MPS not built with PyTorch. Consider rebuilding PyTorch with MPS support if you have an Apple Silicon Mac.")

        logging.info(f"Loading SentenceTransformer model: {model_name}...")
        model = SentenceTransformer(model_name, device=device)

        logging.info("Generating embeddings...")
        embeddings = model.encode(job_descriptions, batch_size=batch_size, convert_to_numpy=True)

        logging.info("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        logging.info(f"Saving FAISS index to {output_index_path}...")
        faiss.write_index(index, output_index_path)

        logging.info("Embeddings and FAISS index creation completed.")

    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {data_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during embeddings creation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/jobs.csv")
    output_index_path = os.path.join(script_dir, "model/job_index.faiss")
    create_embeddings_and_index(data_path, output_index_path)