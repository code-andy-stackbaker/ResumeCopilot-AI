import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from app.classifier.classifier_reranker import ClassifierReranker
import numpy as np
import logging

class JobRecommender:
  def __init__(self, faiss_index_path, metadata_path, model_name="all-MiniLM-L6-v2", top_k=5):
    self.top_k = top_k
    self.model = SentenceTransformer(model_name)
    self.classifier = ClassifierReranker()
    try:
      self.index = faiss.read_index(faiss_index_path)
    except Exception as e:
      logging.error(f"Failed to load FAISS index: {e}")
      raise
    try: 
      self.metadata = pd.read_csv(metadata_path)
    except Exception as e:
      logging.error(f"Failed to load FAISS index: {e}")
      raise

  def recommend(self, resume_text: str):
    print("✅ Starting recommendation...")

    if not resume_text or not isinstance(resume_text, str):
      raise ValueError("Resume text must be a non-empty string.")
    
    try:
      resume_vector = self.model.encode([resume_text], convert_to_numpy=True)
      distances, indices = self.index.search(resume_vector, self.top_k)
    except Exception as e:
      logging.error(f"FAISS search failed: {e}")
      return []
  
    results = []
    print("the indices", indices[0])
    for i, idx in enumerate(indices[0]):
      print(f"🔄 Processing job {i + 1}/{self.top_k} (index={idx})")
      job = self.metadata.iloc[idx]["job_desciption"]
      score = float(distances[0][i])
      print("the score", score)
      print("the job", job)
      print("🧠 Calling classifier reranker...")
      classifier_score = self.classifier.predict_match_score(resume_text, job)
      print(f"✅ Classifier match score: {classifier_score}")
      results.append({
        "rank": i + 1,
        "job_description": job,
        "faiss_score": round(score, 4),
        "classifier_score": round(classifier_score, 4)
      })
    results.sort(key=lambda x: x["classifier_score"], reverse=True)
    return results