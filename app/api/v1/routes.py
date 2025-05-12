from fastapi import APIRouter, HTTPException, status
from app.models.resume import ResumeInput, KeywordOutput
from app.services.keyword_extractor import extract_keywords
from app.recommender.job_recommender import JobRecommender  # Import the recommender
import logging
from pydantic import BaseModel
from typing import List

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

class JobRecommendation(BaseModel):
  rank: int
  job_description: str
  faiss_score: float
  classifier_score: float

@router.post("/generate-keywords", response_model=KeywordOutput)
async def generate_keywords(resume: ResumeInput):
  try: 
    keywords = extract_keywords(resume.resume_text)
    logging.info(f"Extracted keywords: {keywords}")
    return {"keywords" : keywords}
  except Exception as e:
    logging.error(f"Error generating Keywords: {str(e)}", exc_info=True)
    raise HTTPException(status_code= status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to genrate keywords: {e}")
  
  
  
@router.post("/recommend-jobs", response_model=List[JobRecommendation])
async def recommend_jobs(resume: ResumeInput):
  """
  Endpoint to get job recommendations based on a resume.
  """
  logging.info("Received request for job recommendations.")
  try:
    recommender = JobRecommender()  # Instantiate the recommender
    recommendations = recommender.recommend(resume.resume_text)
    logging.info(f"Generated {len(recommendations)} job recommendations.")
    return recommendations
  except Exception as e:
    logging.error(f"Error generating job recommendations: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Failed to generate job recommendations"
    )