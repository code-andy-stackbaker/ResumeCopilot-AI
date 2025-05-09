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
        self._load_components()

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
                config={"max_new_tokens": 350, "temperature": 0.7, "context_length": 2048} # Increased max_new_tokens
            )
            logger.info("LLM loaded successfully.")

            self.prompt_template = PromptTemplate.from_template(
                "Answer the question based on the context below. Be detailed and comprehensive.\n\nContext:\n{context}\n\nQuestion: {input}"
            )

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
        """
        Answers a general question using the FAISS index of all jobs.
        """
        if not self.general_retrieval_chain:
            logger.error("General retrieval chain is not initialized.")
            return {"error": "QAService not properly initialized."}
        try:
            logger.info(f"Answering general question: {question}")
            result = self.general_retrieval_chain.invoke({"input": question})
            return result
        except Exception as e:
            logger.error(f"Error during general question answering: {e}")
            return {"error": str(e)}

    def answer_with_specific_context(self, question: str, context_text: str) -> dict:
        """
        Answers a question based on the specifically provided context text.
        The context_text should already combine relevant information (e.g., resume + job description).
        """
        if not self.combine_docs_chain:
            logger.error("Combine docs chain is not initialized.")
            return {"error": "QAService not properly initialized."}
        try:
            logger.info(f"Answering question with specific context: {question}")
            # The 'create_stuff_documents_chain' expects a list of Document objects for the context
            context_documents = [Document(page_content=context_text)]
            result = self.combine_docs_chain.invoke({
                "input": question,
                "context": context_documents
            })
            return result
        except Exception as e:
            logger.error(f"Error during specific context question answering: {e}")
            return {"error": str(e)}

# Example usage (optional, for testing this file directly)
if __name__ == "__main__":
    try:
        qa_service = QAService()
        
        # Test general Q&A
        general_question = "What skills are common for Python developer roles?"
        print(f"--- Testing General Q&A ---")
        general_answer = qa_service.answer_general_question(general_question)
        print(f"Question: {general_question}")
        if general_answer and "answer" in general_answer:
            print(f"Answer: {general_answer['answer']}\n")
        else:
            print(f"Could not get answer: {general_answer}\n")

        # Test Q&A with specific context
        specific_question = "Why is the candidate with React and Node.js experience a good fit for a job requiring these skills?"
        sample_context = (
            "Resume Snippet: Experienced React developer with backend skills in Node.js and AWS.\n"
            "Job Description: We are looking for a Full Stack Developer proficient in React for frontend and Node.js for backend."
        )
        print(f"--- Testing Q&A with Specific Context ---")
        specific_answer = qa_service.answer_with_specific_context(specific_question, sample_context)
        print(f"Question: {specific_question}")
        if specific_answer and "answer" in specific_answer:
            print(f"Answer: {specific_answer['answer']}")
        else:
            print(f"Could not get answer: {specific_answer}")

    except Exception as e:
        logger.error(f"Failed to run QAService example: {e}")