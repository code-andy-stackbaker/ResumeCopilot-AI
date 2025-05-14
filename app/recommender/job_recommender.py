# app/recommender/job_recommender.py

import logging
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import Recommender_MODEL_NAME # Using this for model name
from app.database.session import SessionLocal
from app.database.models import Job, EMBEDDING_DIMENSION_RECOMMENDER # Import Job model and configured dimension
from app.classifier.classifier_reranker import ClassifierReranker # Ensure this path is correct

# Configure logging (ensure this is appropriate for your project's logging strategy)
# If you have a central logging config (e.g., in main.py or app/utils/logger.py),
# just `logger = logging.getLogger(__name__)` might be enough here.
# For standalone testing of this file, basicConfig might be okay.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JobRecommender:
    def __init__(
        self,
        model_name: str = Recommender_MODEL_NAME,
        top_k_final_output: int = 5,
        top_k_retrieval: int = 20 # Fetch more candidates for re-ranking
    ):
        logger.info(f"Initializing JobRecommender with model: {model_name} for PostgreSQL backend.")
        self.model_name = model_name
        self.sentence_model = None
        self.device = None
        
        self.classifier = ClassifierReranker()
        
        self.top_k_final_output = top_k_final_output
        self.top_k_retrieval = top_k_retrieval

        self._bert_classify_attempt_counter = 0 # For classifier retry simulation if used
        # self._db_query_attempt_counter = 0 # Can add for simulating DB query failures for _get_candidate_jobs...

        self._initialize_components()
        logger.info("JobRecommender initialized successfully using database backend.")

    def _initialize_components(self):
        """Initializes components like the sentence model."""
        logger.info("Initializing JobRecommender components (Sentence Model)...")
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
                logger.info("MPS device is available. Using MPS for SentenceTransformer.")
            else:
                self.device = torch.device("cpu")
                logger.info("MPS device not available or not built with PyTorch. Using CPU for SentenceTransformer.")
                if torch.backends.mps.is_available() and not torch.backends.mps.is_built():
                    logger.warning("MPS is available but not built with PyTorch. MPS support may be limited.")
            
            logger.info(f"Loading SentenceTransformer model: {self.model_name} onto device: {self.device}")
            self.sentence_model = SentenceTransformer(self.model_name, device=str(self.device))
            logger.info("SentenceTransformer model loaded successfully.")

            model_embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
            if model_embedding_dim != EMBEDDING_DIMENSION_RECOMMENDER:
                logger.error(
                    f"CRITICAL MISMATCH: Loaded SentenceTransformer model '{self.model_name}' has an "
                    f"embedding dimension of {model_embedding_dim}, but the database Job model "
                    f"(EMBEDDING_DIMENSION_RECOMMENDER from app.database.models) is configured for {EMBEDDING_DIMENSION_RECOMMENDER} dimensions. "
                    "Vector similarity search will fail or produce incorrect results. "
                    "Please ensure model and database configurations (app/config.py for EMBEDDING_DIMENSION_RECOMMENDER) are consistent."
                )
                raise ValueError("Embedding dimension mismatch between sentence model and DB configuration.")
            else:
                logger.info(f"Embedding dimension consistency check passed: Model and DB configured for {model_embedding_dim} dimensions.")

            logger.info("JobRecommender components (model only) initialized successfully.")

        except Exception as e:
            logger.error(f"Exception during component initialization: {e}", exc_info=True)
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _get_candidate_jobs_from_db_with_retry(self, resume_embedding: np.ndarray):
        """
        Queries PostgreSQL using pgvector for candidate jobs based on resume embedding.
        Retries on failure.
        """
        logger.info("Attempting to get candidate jobs from database with pgvector.")
        
        if not isinstance(resume_embedding, np.ndarray):
            resume_embedding = np.array(resume_embedding) # Ensure it's a numpy array
        
        if resume_embedding.ndim == 1 and resume_embedding.shape[0] != EMBEDDING_DIMENSION_RECOMMENDER:
            logger.error(f"Resume embedding dimension {resume_embedding.shape[0]} does not match DB dimension {EMBEDDING_DIMENSION_RECOMMENDER}")
            raise ValueError("Resume embedding dimension mismatch.")
        elif resume_embedding.ndim > 1: # Should be flattened already, but as a safeguard
             logger.warning(f"Resume embedding has {resume_embedding.ndim} dimensions, expected 1. Attempting to flatten.")
             resume_embedding = resume_embedding.flatten()
             if resume_embedding.shape[0] != EMBEDDING_DIMENSION_RECOMMENDER:
                 logger.error(f"Flattened resume embedding dimension {resume_embedding.shape[0]} does not match DB dimension {EMBEDDING_DIMENSION_RECOMMENDER}")
                 raise ValueError("Flattened resume embedding dimension mismatch.")


        db = None
        try:
            db = SessionLocal()
            
            # Using L2 distance (<->). Lower is better.
            # Alternatives:
            # Cosine distance: Job.job_description_embedding.cosine_distance(resume_embedding) (lower is better, 0 means identical)
            # Max Inner Product: Job.job_description_embedding.max_inner_product(resume_embedding) (higher is better, order by .desc())
            # For normalized embeddings, L2 distance and cosine distance are monotonically related.
            # If your embeddings are normalized, (1 - cosine_distance) or inner_product can act as similarity.
            
            candidate_jobs_with_distance = db.query(
                Job, # Select the whole Job ORM object
                Job.job_description_embedding.l2_distance(resume_embedding).label('distance')
            ).order_by(
                Job.job_description_embedding.l2_distance(resume_embedding) # Order by distance ascending
            ).limit(
                self.top_k_retrieval
            ).all()
            
            logger.info(f"Successfully retrieved {len(candidate_jobs_with_distance)} candidate jobs from database.")
            return candidate_jobs_with_distance
        
        except Exception as e:
            logger.error(f"Error during database query for candidate jobs: {e}", exc_info=True)
            raise 
        finally:
            if db:
                db.close()
                logger.debug("Database session closed after retrieving candidates.")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _predict_match_score_with_retry(self, resume_text: str, job_description: str) -> float: # Added type hint
        """
        Predicts a match score between resume and job description using the classifier.
        Retries on failure.
        """
        # Add a check for empty job_description to prevent errors in classifier
        if not job_description:
            logger.warning("Job description is empty. Classifier cannot process. Returning 0.0 score.")
            return 0.0

        logger.info(f"Attempting classifier prediction for job (description snippet: {job_description[:70]}...).")
        # Simulate failure for BERT classifier if counter is used
        # if self._bert_classify_attempt_counter < 1:
        #     self._bert_classify_attempt_counter += 1
        #     logger.warning(f"Simulating BERT classifier failure, attempt {self._bert_classify_attempt_counter}")
        #     raise Exception("Simulated BERT classifier temporary failure")
        
        try:
            score = self.classifier.predict_match_score(resume_text, job_description)
            logger.info(f"Classifier prediction successful. Score: {score:.4f}")
            # self._bert_classify_attempt_counter = 0 # Reset counter after success if simulating
            return float(score) # Ensure it's a float
        except Exception as e:
            logger.error(f"Error during classifier prediction attempt: {e}", exc_info=True) # Log with full traceback
            raise # Re-raise for Tenacity

    def recommend(self, resume_text: str) -> dict: # Return type to dict for structured output
        """
        Recommends jobs based on the provided resume text, using PostgreSQL+pgvector and a classifier.
        """
        logger.info(f"Starting job recommendation process for resume snippet: '{resume_text[:70]}...'")
        if not resume_text or not isinstance(resume_text, str):
            logger.error("Invalid resume text provided. Must be a non-empty string.")
            # Return a structured error or raise a specific exception
            return {"error": "Invalid resume text.", "recommendations": []}

        if self.sentence_model is None:
            logger.error("Sentence model not initialized. Cannot generate resume embedding.")
            return {"error": "Recommender service not ready (model not loaded).", "recommendations": []}

        try:
            logger.debug("Encoding resume text...")
            # sentence_model.encode() can return a list of embeddings if a list of sentences is passed.
            # For a single string, it usually returns a single ndarray.
            resume_embedding_raw = self.sentence_model.encode(resume_text, convert_to_numpy=True)
            
            # Ensure resume_embedding is a 1D NumPy array of the correct type for pgvector
            if isinstance(resume_embedding_raw, list): # Should not happen with convert_to_numpy=True for single string
                resume_embedding = np.array(resume_embedding_raw[0]).astype(np.float32)
            else: # Assuming it's already a numpy array
                resume_embedding = resume_embedding_raw.astype(np.float32)

            if resume_embedding.ndim > 1: # If it's like [[0.1, 0.2, ...]]
                resume_embedding = resume_embedding.flatten()
            
            logger.debug(f"Resume encoded into vector of dimension {resume_embedding.shape[0]}.")
            if resume_embedding.shape[0] != EMBEDDING_DIMENSION_RECOMMENDER:
                logger.error(f"Resume embedding dimension is {resume_embedding.shape[0]}, expected {EMBEDDING_DIMENSION_RECOMMENDER}")
                return {"error": "Resume embedding dimension mismatch.", "recommendations": []}

        except Exception as e:
            logger.error(f"Error encoding resume text: {e}", exc_info=True)
            return {"error": "Failed to process resume.", "recommendations": []}

        try:
            candidate_jobs_with_scores = self._get_candidate_jobs_from_db_with_retry(resume_embedding)
        except Exception as e:
            logger.error(f"Database search for candidate jobs failed after multiple retries: {e}", exc_info=True)
            return {"error": "Job search functionality is temporarily unavailable.", "recommendations": []}

        if not candidate_jobs_with_scores:
            logger.info("No candidate jobs found from database search.") 
            return {"message": "No initial job matches found.", "recommendations": []}

        ranked_results = []
        logger.info(f"Processing {len(candidate_jobs_with_scores)} candidates from database for re-ranking...")

        for job_object, vector_distance in candidate_jobs_with_scores:
            try:
                job_description = job_object.job_description_text
                if not job_description or not isinstance(job_description, str): # Added check
                    logger.warning(f"Skipping Job ID {job_object.id} ('{job_object.title}') due to missing or invalid description.")
                    continue

                logger.debug(f"Re-ranking job ID {job_object.id} ('{job_object.title[:50]}...') with classifier.")
                classifier_score = self._predict_match_score_with_retry(resume_text, job_description)
                
                ranked_results.append({
                    "job_id": job_object.external_job_id if job_object.external_job_id else f"internal_{job_object.id}",
                    "title": job_object.title,
                    "company": job_object.company,
                    "location": job_object.location,
                    "job_description_snippet": job_description[:200] + "..." if len(job_description) > 200 else job_description,
                    "vector_score_l2_distance": round(float(vector_distance), 6), # Lower is better
                    "classifier_match_score": round(float(classifier_score), 4) # Higher is better
                })
            except Exception as e:
                logger.error(f"Error processing or classifying job ID {job_object.id} ('{job_object.title}'): {e}", exc_info=False) # exc_info=False for less verbose logs in loop
                continue
        
        ranked_results.sort(key=lambda x: x["classifier_match_score"], reverse=True)
        
        final_recommendations = ranked_results[:self.top_k_final_output]
        
        for i, rec in enumerate(final_recommendations):
            rec["final_rank"] = i + 1
            
        logger.info(f"Recommendation process completed. Returning {len(final_recommendations)} recommendations.")
        return {"recommendations": final_recommendations}