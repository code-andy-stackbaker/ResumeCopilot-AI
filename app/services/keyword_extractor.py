from sentence_transformers import SentenceTransformer, util
import torch

# Load the MiniLM model for fast and meaningful sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a fixed list of technical or job-related keywords
CANDIDATE_KEYWORDS = [
    "Python", "React", "Node.js", "MongoDB", "SQL", "AWS", "Docker",
    "CI/CD", "REST APIs", "Microservices", "TensorFlow", "PyTorch",
    "FastAPI", "GraphQL", "Scrum", "Agile", "Redis", "Kafka"
]

@torch.no_grad()
def extract_keywords(resume_text: str, top_k: int = 8) -> str:
    """
    Extract the top-k relevant keywords from resume_text by comparing
    semantic similarity between resume and known keywords.
    """

    # Step 1: Generate a dense embedding for the full resume text
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    # Step 2: Generate embeddings for all predefined candidate keywords
    keyword_embeddings = model.encode(CANDIDATE_KEYWORDS, convert_to_tensor=True)

    # Step 3: Compute cosine similarity between resume and each keyword
    # This gives us a tensor of similarity scores for each keyword
    cosine_scores = util.cos_sim(resume_embedding, keyword_embeddings)[0]

    # Step 4: Select the top-k highest scoring keywords
    top_results = torch.topk(cosine_scores, k=top_k)

    # Step 5: Map the indices of top keywords to their string names
    top_keywords = [CANDIDATE_KEYWORDS[i] for i in top_results.indices]

    # Step 6: Return the selected keywords as a comma-separated string
    return ", ".join(top_keywords)