from fastapi import APIRouter
from app.models.resume import ResumeInput, KeywordOutput
from app.services.keyword_extractor import extract_keywords

router = APIRouter()

@router.post("/generate-keywords", response_model=KeywordOutput)
async def generate_keywords(resume: ResumeInput):
    keywords = extract_keywords(resume.resume_text)
    return {"keywords": keywords}