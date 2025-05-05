
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers  # or use a mock LLM if no API key
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  

# langchain_qa/
FAISS_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

# Load FAISS index and embedder
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DIR, embedder, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()


llm = CTransformers(
  model="/Volumes/Data/Development/Programming/Python/LLM/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
  model_type="mistral",
  config={
      "max_new_tokens": 256,
      "temperature": 0.7,
      "context_length": 2048,
  }
)


prompt = PromptTemplate.from_template(
    "Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {input}"
)

# ✅ Create retrieval Q&A chain
combine_docs_chain = create_stuff_documents_chain(
    llm, prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ✅ Ask a question
question = "What skills are needed for a Python developer?"
result = retrieval_chain.invoke({"input": question})

# ✅ Output result
print("🔎 Question:", question)
print("🧪 Raw result:", result)
print("✅ Answer:", result["answer"])