from fastapi import APIRouter, HTTPException, status
from app.models.resume import ResumeInput, KeywordOutput

from app.services.keyword_extractor import extract_keywords
from app.recommender.job_recommender import JobRecommender  # Import the recommender
import logging
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional
from app.langchain_qa.services import QAService


#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

class JobItemOutputSchema(BaseModel): # Example - should match what recommender outputs
    job_id: str
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    job_description_snippet: Optional[str] = None
    vector_score_l2_distance: float
    classifier_match_score: float
    final_rank: int
    class Config:
      from_attributes = True # For Pydantic v2+ (orm_mode is for v1)


class JobRecommendationResponseSchema(BaseModel):
    recommendations: List[JobItemOutputSchema]
    class Config:
        from_attributes = True
  
class QAInput(BaseModel):
    query: str
    context_text: str = None  # Optional context
    resume: str = None

class QAResponse(BaseModel):
  answer: str
  source_documents: List[Dict] = []  # Adjust based on your actual source_documents  

@router.post("/generate-keywords", response_model=KeywordOutput)
async def generate_keywords(resume: ResumeInput):
  try: 
    keywords = extract_keywords(resume.resume_text)
    logging.info(f"Extracted keywords: {keywords}")
    return {"keywords" : keywords}
  except Exception as e:
    logging.error(f"Error generating Keywords: {str(e)}", exc_info=True)
    raise HTTPException(status_code= status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to genrate keywords: {e}")
  
  
  
@router.post("/recommend-jobs", response_model= JobRecommendationResponseSchema)
async def recommend_jobs(resume: ResumeInput):
  """
  Endpoint to get job recommendations based on a resume.
  """
  logging.info("Received request for job recommendations.")
  try:
    recommender = JobRecommender()  # Instantiate the recommender
    recommendations = recommender.recommend(resume.resume_text)
    logging.info(f"Generated {len(recommendations)} job recommendations.", exc_info=True)
    return recommendations
  except ValidationError as ve:  # Handle Pydantic validation errors
    logging.error(f"Validation error: {ve}", contentFetchId='uploaded:code-andy-stackbaker/ResumeCopilot-AI/ResumeCopilot-AI-c028b77728b4c051090ae3bdc293a8597a35b572/app/api/v1/routes.py')
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,  # Use 422 for validation errors
        detail=ve.errors()
    )
  except Exception as e:
    logging.error(f"Error generating job recommendations: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Failed to generate job recommendations : {e}"
    )
    
@router.post("/qa/general", response_model=QAResponse)
async def qa_general(qa_input: QAInput):
  """
  Endpoint for general knowledge base queries.
  """
  logging.info(f"Received general QA query: {qa_input.query}")
  try:
      qa_service = QAService()
      result = qa_service.answer_general_question(qa_input.query)
      logging.info(f"Generated general QA answer: {result}")
      return {"answer": result["answer"], "source_documents": []}  # Extract answer, adjust source_documents
  except Exception as e:
      logging.error(f"Error processing general QA query: {e}", exc_info=True)
      raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"Failed to process general QA query: {str(e)}"
      )

@router.post("/qa/context", response_model=QAResponse)
async def qa_context(qa_input: QAInput):
  """
  Endpoint for QA with specific context (e.g., resume + job description).
  """
  
  context_for_llm = f"Candidate's Resume:\n{qa_input.resume}\n\nSpecific Job Description for Analysis:\n{qa_input.context_text}"
  
  logging.info(f"Received context-based QA query: {qa_input.query}")
  if not qa_input.context_text:
      raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="context_text is required for context-based QA"
      )
  try:
      qa_service = QAService()
      result = qa_service.answer_with_specific_context(qa_input.query, context_for_llm)
      logging.info(f"Generated context-based QA answer: {result}")
      return {"answer": result['answer']} # adjust based on your actual data
  except Exception as e:
      logging.error(f"Error processing context-based QA query: {e}", exc_info=True)
      raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"Failed to process context-based QA query: {str(e)}"
      )
    