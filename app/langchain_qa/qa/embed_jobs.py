import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/jobs.csv")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "../embeddings/faiss_index")

df = pd.read_csv(DATA_PATH)

documents = []
for i, doc in df.iterrows():
  doc = Document( page_content=doc["job_description"], metadata = {"job_id": int(doc["job_id"])})
  documents.append(doc)
  
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


