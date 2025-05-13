# app/langchain_qa/services.py

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema.document import Document
import torch # For MPS check
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log # New import


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
# Path to the FAISS index created by app/langchain_qa/qa/embed_jobs.py
# Assumes services.py is in app/langchain_qa/
# So, "qa/embeddings" is a subdirectory relative to this file's parent directory.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_QA_INDEX_DIR = os.path.join(CURRENT_DIR, "qa", "embeddings")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# IMPORTANT: Path to your local LLM model file.
# This was taken from your app/langchain_qa/qa/run_qa.py.
# Ensure this path is correct for your environment.
LLM_MODEL_PATH = "/Volumes/Data/Development/Programming/Python/LLM/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
LLM_MODEL_TYPE = "mistral"

class QAService:
    def __init__(self):
      self.embedder = None
      self.vectorstore = None
      self.retriever = None
      self.llm = None
      self.prompt_template = None
      self.combine_docs_chain = None
      self.general_retrieval_chain = None
      retrieval_chain = None
      
      
      if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        self.device = torch.device("mps")
        logger.info("MPS device is available. Using MPS for SentenceTransformer.")
      else:
        self.device = torch.device("cpu")
        logger.info("MPS device not available. Using CPU for SentenceTransformer.")
        if not torch.backends.mps.is_built():
          logger.warning("MPS not built with PyTorch. Consider rebuilding PyTorch with MPS support if you have an Apple Silicon Mac.")
      
      self._load_components()
      
    # Inside the QAService class:

    @retry(
      wait=wait_exponential(multiplier=1, min=1, max=10), # Wait 1s, 2s, 4s, 8s, then 10s (longer max for potentially slower LLMs)
      stop=stop_after_attempt(3),
      before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _invoke_general_retrieval_chain_with_retry(self, question: str):
      logger.info(f"Attempting to invoke general retrieval chain for question: {question[:70]}...")
      if not self.general_retrieval_chain:
        logger.error("General retrieval chain is not initialized before invoking.")
        # This is a programming error / state issue, not something retries will fix.
        raise SystemError("QAService general retrieval chain not initialized.")
      
      try:
        result = self.general_retrieval_chain.invoke({"input": question})
        logger.info("General retrieval chain invoked successfully.")
        return result
      except Exception as e:
        logger.error(f"Error during general_retrieval_chain.invoke attempt: {e}", exc_info=False)
        raise # Re-raise for Tenacity to handle
      
    # Inside the QAService class:

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _invoke_combine_docs_chain_with_retry(self, question: str, context_documents: list[Document]):
      logger.info(f"Attempting to invoke combine docs chain for question: {question[:70]}...")
      if not self.combine_docs_chain:
        logger.error("Combine docs chain is not initialized before invoking.")
        raise SystemError("QAService combine docs chain not initialized.")
      try:
        # The context_documents are already prepared as a list of Document objects
        raw_result = self.combine_docs_chain.invoke({
            "input": question,
            "context": context_documents
        })
        logger.info("Combine docs chain invoked successfully.")
        return raw_result
      except Exception as e:
        logger.error(f"Error during combine_docs_chain.invoke attempt: {e}", exc_info=False)
        raise # Re-raise for Tenacity    

    def _load_components(self):
      try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        logger.info(f"Loading FAISS index from: {FAISS_QA_INDEX_DIR}")
        if not os.path.exists(FAISS_QA_INDEX_DIR):
          raise FileNotFoundError(f"FAISS index directory not found: {FAISS_QA_INDEX_DIR}. Please ensure 'app/langchain_qa/qa/embed_jobs.py' has been run successfully.")
        self.vectorstore = FAISS.load_local(
          FAISS_QA_INDEX_DIR,
          self.embedder,
          allow_dangerous_deserialization=True # Make sure you trust the source of the FAISS index
        )
        self.retriever = self.vectorstore.as_retriever()
        logger.info("FAISS index and retriever loaded successfully for general Q&A.")

        logger.info(f"Loading LLM model from: {LLM_MODEL_PATH}")
        self.llm = CTransformers(
          model=LLM_MODEL_PATH,
          model_type=LLM_MODEL_TYPE,
          config={"max_new_tokens": 350, "temperature": 0.7, "context_length": 2048, "gpu_layers": 20 } # Increased max_new_tokens
        )
        logger.info("LLM loaded successfully.")
        
        # Inside _load_components method, replace the old prompt_template line with this:
        detailed_prompt_text = (
          "Use the following pieces of context ONLY to answer the question at the end. "
          "The context may contain a candidate's resume and a specific job description. "
          "If the question asks about required skills for the job, focus ONLY on the 'Specific Job Description' part of the context. "
          "If the context does not contain the answer, clearly state that the answer cannot be determined from the provided context.\n\n"
          "Context:\n{context}\n\n"
          "Question: {input}\n"
          "Answer:"
        )
        self.prompt_template = PromptTemplate.from_template(detailed_prompt_text)
        logger.info("Using detailed prompt template.")

        # This chain is for answering questions with provided context (e.g., specific job + resume)
        self.combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt_template)

        # This chain is for general Q&A using the retriever over the whole job database
        self.general_retrieval_chain = create_retrieval_chain(self.retriever, self.combine_docs_chain)
        logger.info("Q&A chains created successfully.")

      except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError during QAService initialization: {fnf_error}")
        raise  # Re-raise to signal failure to initialize
      except Exception as e:
        logger.error(f"Error loading Q&A components: {e}")
        # Depending on desired robustness, you might raise e or handle it by setting components to None
        raise # Re-raise to signal failure to initialize

    def answer_general_question(self, question: str) -> dict:
      if not self.general_retrieval_chain:
        logger.error("General retrieval chain is not initialized.")
        return {"error": "QAService not properly initialized."}
      try:
        logger.info(f"Answering general question: {question}")
        result = self._invoke_general_retrieval_chain_with_retry(question)
        return result
      except Exception as e:
        logger.error(f"Error during general question answering: {e}")
        return {"error": str(e)}

    def answer_with_specific_context(self, question: str, context_text: str) -> dict:
      if not self.combine_docs_chain:
        logger.error("Combine docs chain is not initialized.")
        return {"error": "QAService not properly initialized."}
      try:
        logger.info(f"Answering question with specific context: {question}")
        # The 'create_stuff_documents_chain' expects a list of Document objects for the context
        context_documents = [Document(page_content=context_text)]
        logger.info(f"Context documents being passed: {context_documents}") # Log the input docs

        logger.info(f"Invoking combine_docs_chain...")
        raw_result = self._invoke_combine_docs_chain_with_retry(question, context_text)
        logger.info(f"RAW result dictionary from combine_docs_chain:") # <-- Log the raw output

        # Check if the expected 'answer' key exists and has content
        if raw_result and isinstance(raw_result, str):
          return raw_result # Return the full dictionary if valid
        else:
          logger.error(f"LLM chain did not return a valid answer dictionary or answer was empty. Raw result: {raw_result}")
          # Return a dictionary indicating the issue, potentially including the raw result if helpful
          return {"error": "LLM did not provide a valid answer in the expected format.", "raw_result": raw_result}

      except Exception as e:
        # Log the exception with traceback for better debugging
        logger.error(f"Error during specific context question answering: {e}", exc_info=True)
        return {"error": str(e)}

