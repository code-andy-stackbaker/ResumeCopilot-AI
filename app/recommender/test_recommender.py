import torch.multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

from app.recommender.job_recommender import JobRecommender
import os 

# Define file paths
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "model/job_index.faiss") 
METADATA_PATH = os.path.join(os.path.dirname(__file__), "model/job_metadata.csv")

# Create instance of recommender
recommender = JobRecommender(
    faiss_index_path=FAISS_INDEX_PATH,
    metadata_path=METADATA_PATH,
    top_k=5
)

# Sample resume input
resume_text = "Experienced React developer with backend skills in Node.js and AWS."

# Get recommendations
recommendations = recommender.recommend(resume_text)

# Print results
print("\n🔍 Top job matches (with FAISS + Classifier reranking):\n")
for job in recommendations:
  print(f"{job['rank']}. {job['job_description']}")
  print(f"   🔗 FAISS Score: {job['faiss_score']}")
  print(f"   🧠 Classifier Match Score: {job['classifier_score']}\n")