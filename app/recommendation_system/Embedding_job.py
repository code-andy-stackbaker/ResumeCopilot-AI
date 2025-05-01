import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

df = pd.read_csv("resume_Job_Match_Dataset.csv")
job_texts = df["job_description"].drop_duplicates().to_list()
model = SentenceTransformer("all-MiniLM-L6-v2")


job_embeddings = model.encode(job_texts, convert_to_numpy=True, show_progress_bar=True)
dimension = job_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(job_embeddings)


faiss.write_index(index, "job_index.faiss")
pd.DataFrame({
  "job_id": range(len(job_texts)),
  "job_desciption": job_texts
}).to_csv("job_metadata.csv", index=False)



print("Done...")