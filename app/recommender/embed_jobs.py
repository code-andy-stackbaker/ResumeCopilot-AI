import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def build_faiss_index_from_csv(
    csv_path: str,
    index_path: str,
    metadata_path: str,
    text_column: str = "job_description",
    model_name: str = "all-MiniLM-L6-v2"
):
    # Step 1: Load dataset
    print("📂 Loading job data from CSV...")
    df = pd.read_csv(csv_path)
    job_texts = df[text_column].drop_duplicates().to_list()

    # Step 2: Load embedding model
    print(f"🧠 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Step 3: Encode job descriptions
    print(f"🔄 Encoding {len(job_texts)} job descriptions...")
    job_embeddings = model.encode(job_texts, convert_to_numpy=True, show_progress_bar=True)

    # Step 4: Build FAISS index
    print("📦 Creating FAISS index...")
    dimension = job_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(job_embeddings)

    # Step 5: Save index and metadata
    print("💾 Saving index and metadata...")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    pd.DataFrame({
        "job_id": range(len(job_texts)),
        "job_desciption": job_texts
    }).to_csv(metadata_path, index=False)

    print("✅ Done — FAISS index and metadata saved.")


if __name__ == "__main__":
    build_faiss_index_from_csv(
        csv_path="resume_Job_Match_Dataset.csv",
        index_path="recommender/model/job_index.faiss",
        metadata_path="recommender/model/job_metadata.csv"
    )