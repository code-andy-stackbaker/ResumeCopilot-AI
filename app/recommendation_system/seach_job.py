import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load FAISS index and metadata
index = faiss.read_index("job_index.faiss")
job_metadata = pd.read_csv("job_metadata.csv")

# Step 2: Load the same sentence-transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Define a new resume (user input)
resume_text = "Senior React developer with backend experience in Node.js, MongoDB, and AWS."

# Step 4: Embed the resume
resume_vector = model.encode([resume_text], convert_to_numpy=True)

# Step 5: Search FAISS index for top 5 closest jobs
top_k = 5
distances, indices = index.search(resume_vector, top_k)

# Step 6: Fetch and display matching job descriptions
print("🔍 Top matching jobs for the resume:\n")
for i, idx in enumerate(indices[0]):
    job = job_metadata.iloc[idx]["job_desciption"]
    score = distances[0][i]
    print(f"{i+1}. {job}\n   🔗 FAISS Distance Score: {score:.4f}\n")