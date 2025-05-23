# app/langchain_qa/services.py

import os
import logging
import numpy as np
import torch
from typing import List, Dict # Ensure List and Dict are imported

# Langchain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.document import Document

# Tenacity for retries
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

# Application-specific imports
# Assuming config.py is in app directory and .env is in project root
# Ensure sys.path is handled correctly if running scripts from different locations
try:
    # These imports are essential for the QAService to function with the database
    from app.config import DATABASE_URL, EMBEDDING_DIMENSION_RECOMMENDER # EMBEDDING_DIMENSION_RECOMMENDER is used for QA_EMBEDDING_DIMENSION default
    from app.database.session import SessionLocal
    from app.database.models import Job # Job model is crucial for querying with pgvector
except ImportError:
    # Fallback for standalone execution or if path issues occur during direct script run
    # This indicates a potential issue with how the script is run or PYTHONPATH
    logging.error("Failed to import from app.config or app.database. Make sure PYTHONPATH is set correctly or script is run as part of the app.")
    # Define fallbacks if necessary for the script to be syntactically valid, though it might not function.
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:pass@host:port/db")
    EMBEDDING_DIMENSION_RECOMMENDER = int(os.getenv("EMBEDDING_DIMENSION_RECOMMENDER", 384))
    # SessionLocal and Job would need mock definitions or the script would fail later
    # This is primarily to allow the script to be parsed if imports fail in certain contexts.
    # Proper execution requires the app modules to be importable.
    class SessionLocal: pass # Placeholder
    class Job: pass # Placeholder


# Setup logging
logger = logging.getLogger(__name__)
# Ensure logging is configured once at your application's entry point (e.g., main.py)
# For example: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Configuration ---
QA_EMBEDDING_MODEL_NAME = os.getenv("QA_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# QA_EMBEDDING_DIMENSION should match the embeddings in the DB (Job.job_description_embedding)
# It defaults to EMBEDDING_DIMENSION_RECOMMENDER from config, which should be consistent with your ingested embeddings.
QA_EMBEDDING_DIMENSION = int(os.getenv("QA_EMBEDDING_DIMENSION", EMBEDDING_DIMENSION_RECOMMENDER))

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH") # Load from .env
LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE") # Load from .env
NUM_DOCS_TO_RETRIEVE_FOR_QA = int(os.getenv("NUM_DOCS_TO_RETRIEVE_FOR_QA"))
logger.info("the total number fetch :", NUM_DOCS_TO_RETRIEVE_FOR_QA)


class QAService:
    def __init__(self):
        self.embedder = None
        self.llm = None
        self.prompt_template = None
        self.combine_docs_chain = None
        self.device = None
        self.general_retrieval_chain = None # This attribute exists but is not used in the answer_general_question flow below

        self._determine_device()
        self._load_components()
        logger.info("QAService initialized. Context retrieval will use PostgreSQL with pgvector.")

    def _determine_device(self):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            logger.info("MPS device is available. Using MPS for HuggingFaceEmbeddings.")
        else:
            self.device = torch.device("cpu")
            logger.info("MPS device not available or not built with PyTorch. Using CPU for HuggingFaceEmbeddings.")
            if torch.backends.mps.is_available() and not torch.backends.mps.is_built(): # Check if MPS is available but not built
                logger.warning("MPS is available but PyTorch was not built with MPS support. MPS support may be limited.")

    def _load_components(self):
        try:
            logger.info(f"Loading Q&A embedding model: {QA_EMBEDDING_MODEL_NAME}")
            self.embedder = HuggingFaceEmbeddings(
                model_name=QA_EMBEDDING_MODEL_NAME,
                model_kwargs={'device': str(self.device)} # Pass device as string
            )
            logger.info("Q&A Embedding model object created. Determining its dimension...")

            # Robustly get embedding dimension
            try:
                dummy_text = "get dimension" # A short string to embed
                dummy_embedding = self.embedder.embed_query(dummy_text)
                model_embed_dim = len(dummy_embedding)
                logger.info(f"Successfully determined Q&A embedding model '{QA_EMBEDDING_MODEL_NAME}' dimension as {model_embed_dim}.")
            except Exception as e_dim:
                logger.error(f"Failed to determine embedding dimension for Q&A model '{QA_EMBEDDING_MODEL_NAME}'. Error: {e_dim}", exc_info=True)
                raise ValueError(f"Could not determine embedding dimension for Q&A model: {QA_EMBEDDING_MODEL_NAME}") from e_dim

            # Perform the dimension consistency check
            if model_embed_dim != QA_EMBEDDING_DIMENSION:
                error_msg = (f"CRITICAL MISMATCH: Q&A Embedding model '{QA_EMBEDDING_MODEL_NAME}' (actual dimension: {model_embed_dim}) "
                             f"differs from configured QA_EMBEDDING_DIMENSION ({QA_EMBEDDING_DIMENSION}). ")
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Q&A Embedding model dimension ({model_embed_dim}) matches configured QA_EMBEDDING_DIMENSION ({QA_EMBEDDING_DIMENSION}). This is crucial for pgvector queries.")
            logger.info("Q&A Embedding model loaded and dimension verified successfully.")

            if not os.path.exists(LLM_MODEL_PATH): # Check if LLM model file exists
                logger.error(f"LLM model file not found at configured LLM_MODEL_PATH: {LLM_MODEL_PATH}")
                raise FileNotFoundError(f"LLM model file not found: {LLM_MODEL_PATH}")
            
            llm_config = {"max_new_tokens": 350, "temperature": 0.3, "context_length": 4096}
           
            self.llm = CTransformers(
                model=LLM_MODEL_PATH,
                model_type=LLM_MODEL_TYPE,
                config=llm_config
            )
            logger.info("LLM loaded successfully.")
            
            detailed_prompt_text = (
                "Based on the following job description context, please answer the question. "
                "Your answer should be derived solely from the information found within this provided context. "
                "If the context directly answers the question, provide that answer. "
                "If the context does not directly answer the question but contains related information, summarize what relevant information is present. "
                "Avoid making assumptions or using external knowledge.\n\n"
                "Context:\n{context}\n\n"
                "Question: {input}\n"
                "Concise Answer from Context:"
            )
            self.prompt_template = PromptTemplate.from_template(detailed_prompt_text)
            logger.info("Using detailed prompt template for Q&A.")

            self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
            logger.info("Q&A combine_docs_chain created successfully.")

        except FileNotFoundError as fnf_error:
            logger.error(f"FileNotFoundError during QAService component initialization: {fnf_error}")
            raise
        except Exception as e:
            logger.error(f"Error loading Q&A components: {e}", exc_info=True)
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _retrieve_job_contexts_from_db(self, question_embedding: np.ndarray, top_k: int = NUM_DOCS_TO_RETRIEVE_FOR_QA) -> List[Document]:
        logger.info(f"Retrieving top {top_k} job contexts from database using pgvector for question embedding.")
        
        if not isinstance(question_embedding, np.ndarray):
            question_embedding = np.array(question_embedding)
        # pgvector expects a 1D array for the vector being compared against
        if question_embedding.ndim > 1: 
            question_embedding = question_embedding.flatten()
        
        # Ensure the dimension of the question embedding matches the configured dimension for queries
        if question_embedding.shape[0] != QA_EMBEDDING_DIMENSION:
            logger.error(f"Question embedding dimension {question_embedding.shape[0]} does not match expected DB query dimension {QA_EMBEDDING_DIMENSION}.")
            raise ValueError("Question embedding dimension mismatch for DB query.")

        db = None
        retrieved_documents = []
        try:
            db = SessionLocal()
            # This is the core pgvector integration:
            # Querying the 'jobs' table and ordering by the L2 distance (similarity)
            # between the job_description_embedding (stored via pgvector) and the question_embedding.
            results = db.query(
                Job.job_description_text, 
                Job.title,                
                Job.company,              
                Job.location,             
                Job.external_job_id,
                Job.id # Include internal DB ID for a reliable fallback source_id
            ).order_by(
                Job.job_description_embedding.l2_distance(question_embedding) # PGVECTOR SIMILARITY SEARCH
            ).limit(top_k).all()

            logger.info(f"Retrieved {len(results)} potential contexts from database via pgvector.")

            for row_data in results: 
                doc_metadata = {
                    "title": row_data.title,
                    "company": row_data.company,
                    "location": row_data.location,
                    "source_id": row_data.external_job_id if row_data.external_job_id else f"db_job_id_{row_data.id}"
                }
                doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}

                doc = Document(
                    page_content=str(row_data.job_description_text)
                )
                retrieved_documents.append(doc)
            
            return retrieved_documents
        except Exception as e:
            # If this is an OperationalError due to "transaction aborted",
            # it means a previous error (like table not existing or pgvector extension not enabled) occurred.
            logger.error(f"Error during database retrieval for Q&A contexts (pgvector): {e}", exc_info=True)
            # Re-raise to be caught by the main answer_general_question method's error handling
            raise 
        finally:
            if db:
                db.close()
                logger.debug("Database session closed after retrieving Q&A contexts.")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _invoke_combine_docs_chain_with_retry(self, question: str, context_documents: List[Document]) -> str:
        logger.info(f"Attempting to invoke combine docs chain for question: '{question[:10]}...' with {len(context_documents)} context docs.")
        print("the context documents", context_documents)
        if not self.combine_docs_chain:
            logger.error("Combine docs chain is not initialized before invoking.")
            raise SystemError("QAService combine docs chain not initialized.")
        
        if not context_documents: 
            logger.warning("No context documents provided to combine_docs_chain. LLM will rely on its general knowledge or state no context as per prompt.")
            # We still pass empty list to the chain; the prompt instructs LLM how to handle no context.

        try:
            answer_string = self.combine_docs_chain.invoke({
                "input": question,
                "context": context_documents 
            })
            logger.info("Combine docs chain invoked successfully.", answer_string)
            return str(answer_string) if answer_string is not None else ""
        except Exception as e:
            logger.error(f"Error during combine_docs_chain.invoke attempt: {e}", exc_info=True) 
            raise
      
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )

        
    def answer_general_question(self, question: str) -> Dict:
        logger.info(f"Answering general question: '{question[:100]}...' using pgvector retrieval from database.")
        if not all([self.embedder, self.llm, self.combine_docs_chain]):
            logger.error("QAService is not properly initialized (embedder, LLM, or chain missing).")
            return {"error": "QAService not properly initialized.", "answer": "Service is not ready."}

        try:
            logger.debug("Embedding the user question for database retrieval...")
            question_embedding_list = self.embedder.embed_query(question) 
            question_embedding_np = np.array(question_embedding_list).astype(np.float32)
            logger.debug("Question embedded successfully.")

            # Retrieve contexts using pgvector
            context_documents = self._retrieve_job_contexts_from_db(question_embedding_np)
            # The print statements below are for debugging, you might want to remove or conditionalize them for production.
            # print("the context_documentss", context_documents) 

            if not context_documents:
                logger.info("No relevant contexts found in the database for the question via pgvector. Proceeding to LLM without specific DB context.")
            
            answer_string = self._invoke_combine_docs_chain_with_retry(question, context_documents)
            # print("the answer string", answer_string) # Debugging print

            if answer_string and answer_string.strip():
                return {"answer": answer_string.strip(), "retrieved_context_count": len(context_documents)}
            else:
                logger.warning(f"LLM chain returned an empty or invalid answer string. Raw result: '{answer_string}'")
                return {"error": "LLM did not provide a valid answer.", "answer": "I could not formulate an answer for your question at this time.", "retrieved_context_count": len(context_documents)}

        except Exception as e:
            logger.error(f"Error during general question answering after retries: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}", "answer": "Sorry, I encountered an issue while trying to answer."}

    def answer_with_specific_context(self, question: str, context_text: str) -> Dict:
        logger.info(f"Answering question with specific user-provided context: '{question[:70]}...'")
        if not self.combine_docs_chain or not self.llm:
            logger.error("Combine docs chain or LLM not initialized.")
            return {"error": "QAService not properly initialized.", "answer":"Service not ready."}
        try:
            context_documents = [Document(page_content=context_text)]
            
            answer_string = self._invoke_combine_docs_chain_with_retry(question, context_documents)
            
            if answer_string and isinstance(answer_string, str) and answer_string.strip():
                return {"answer": answer_string.strip()}
            else:
                logger.error(f"LLM chain did not return a valid answer string from specific context. Raw result: {answer_string}")
                return {"error": "LLM did not provide a valid answer from the provided context.", "answer": "I could not process that request with the given context.", "raw_result": str(answer_string)}

        except Exception as e:
            logger.error(f"Error during specific context question answering after retries: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {str(e)}", "answer": "Sorry, I encountered an issue while trying to process your request with specific context."}

