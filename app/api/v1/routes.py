from fastapi import APIRouter, HTTPException, status
from app.models.resume import ResumeInput, KeywordOutput
from app.services.keyword_extractor import extract_keywords
import logging

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

@router.post("/generate-keywords", response_model=KeywordOutput)
async def generate_keywords(resume: ResumeInput):
  try: 
    keywords = extract_keywords(resume.resume_text)
    logging.info(f"Extracted keywords: {keywords}")
    return {"keywords" : keywords}
  except Exception as e:
    logging.error(f"Error generating Keywords: {str(e)}", exc_info=True)
    raise HTTPException(status_code= status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to genrate keywords: {e}")