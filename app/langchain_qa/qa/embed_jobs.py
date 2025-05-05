import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/jobs.csv")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "embeddings")

df = pd.read_csv(DATA_PATH)
documents = []
print("the data", df)
print("the index path", INDEX_PATH)

for i, doc in df.iterrows():
  doc = Document( page_content=doc["job_description"], metadata = {"job_id": int(doc["job_id"])})
  documents.append(doc)
  
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

os.makedirs(INDEX_PATH, exist_ok=True)
vectorstore.save_local(INDEX_PATH)