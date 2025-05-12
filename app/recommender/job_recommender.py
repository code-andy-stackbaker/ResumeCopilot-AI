import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from app.classifier.classifier_reranker import ClassifierReranker
import logging
import torch
import os
from app.config import Recommender_MODEL_NAME, FAISS_INDEX_PATH, METADATA_PATH  # Import config variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JobRecommender:
    def __init__(
        self,
        faiss_index_path: str = FAISS_INDEX_PATH,  # Use config
        metadata_path: str = METADATA_PATH,  # Use config
        model_name: str = Recommender_MODEL_NAME,  # Use config
        top_k: int = 5
    ):
        self.top_k = top_k
        self.classifier = ClassifierReranker()
        self.model_name = model_name
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.model = None
        self.index = None
        self.metadata = None
        self.device = None
        self._initialize_components()

    def _initialize_components(self):
        try:
            # Determine device for SentenceTransformer
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
              self.device = torch.device("mps")
              logging.info("MPS device is available. Using MPS for SentenceTransformer.")
            else:
              self.device = torch.device("cpu")
              logging.info("MPS device not available. Using CPU for SentenceTransformer.")
              if not torch.backends.mps.is_built():
                logging.warning("MPS not built with PyTorch. Consider rebuilding PyTorch with MPS support if you have an Apple Silicon Mac.")
            self.model = SentenceTransformer(self.model_name, device=self.device)

            if not os.path.exists(self.faiss_index_path):
                raise FileNotFoundError(f"FAISS index file not found at: {self.faiss_index_path}")
            self.index = faiss.read_index(self.faiss_index_path)

            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")
            self.metadata = pd.read_csv(self.metadata_path)

            logging.info("JobRecommender components initialized successfully.")

        except FileNotFoundError as fnf_error:
            logging.error(f"FileNotFoundError during initialization: {fnf_error}")
            raise
        except Exception as e:
            logging.error(f"Exception during initialization: {e}", exc_info=True)
            raise

    def recommend(self, resume_text: str) -> list:
        """
        Recommends jobs based on the provided resume text.
        """
        logging.info("Starting job recommendation process.")
        if not resume_text or not isinstance(resume_text, str):
            raise ValueError("Resume text must be a non-empty string.")

        try:
            resume_vector = self.model.encode([resume_text], convert_to_numpy=True)
            distances, indices = self.index.search(resume_vector, self.top_k)
        except Exception as e:
            logging.error(f"Error during FAISS search: {e}", exc_info=True)
            return []

        results = []
        for i in range(indices.shape[1]):
            job_description = self.metadata.iloc[indices[0, i]]["job_desciption"]
            faiss_score = float(distances[0, i])
            logging.info(f"Processing job {i + 1}/{self.top_k}: {job_description[:50]}...")
            try:
                classifier_score = self.classifier.predict_match_score(resume_text, job_description)
                logging.info(f"Classifier match score: {classifier_score}")
            except Exception as e:
                logging.error(f"Error during classifier prediction: {e}", exc_info=True)
                classifier_score = 0.0  # Or handle this appropriately

            results.append({
                "rank": i + 1,
                "job_description": job_description,
                "faiss_score": round(faiss_score, 4),
                "classifier_score": round(classifier_score, 4)
            })

        results.sort(key=lambda x: x["classifier_score"], reverse=True)
        logging.info("Recommendation process completed successfully.")
        return results