import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Load model (MiniLM — small and powerful)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load CSV dataset
df = pd.read_csv("/Volumes/Data/Development/Programming/Python/Basic_Fundamentals/CTR_Model/ctr_dataset.csv")

resumes = df["resume_text"].to_list()
job_description = df["job_description"].tolist()
labels = df["clicked"].tolist()

resume_embedding = model.encode(resumes, convert_to_tensor = True)
job_embedding = model.encode(job_description, convert_to_tensor=True)

# Dim = 1 Concatenate resume and job embeddings horizontally to form a combined feature vector for each sample.
# This allows the model to learn from both the resume and job description together (dim=1 means side-by-side).
X = torch.cat( [resume_embedding, job_embedding], dim = 1 )

y = torch.tensor(labels).float()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

torch.save((X_train, y_train), "/Volumes/Data/Development/Programming/Python/Basic_Fundamentals/CTR_Model/ctr_train.pt")
torch.save((X_val, y_val), "/Volumes/Data/Development/Programming/Python/Basic_Fundamentals/CTR_Model/ctr_val.pt")

print("resume_embedding", resume_embedding.size())