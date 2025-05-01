import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class JobRecommender:
    def __init__(self, faiss_index_path, metadata_path, model_name="all-MiniLM-L6-v2", top_k=5):
        self.top_k = top_k
        self.index = faiss.read_index(faiss_index_path)
        self.metadata = pd.read_csv(metadata_path)
        self.model = SentenceTransformer(model_name)

    def recommend(self, resume_text: str):
        if not resume_text or not isinstance(resume_text, str):
            raise ValueError("Resume text must be a non-empty string.")

        resume_vector = self.model.encode([resume_text], convert_to_numpy=True)
        distances, indices = self.index.search(resume_vector, self.top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            job = self.metadata.iloc[idx]["job_desciption"]
            score = float(distances[0][i])
            results.append({
                "rank": i + 1,
                "job_description": job,
                "faiss_score": round(score, 4)
            })

        return results