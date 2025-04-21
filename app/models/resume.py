from pydantic import BaseModel

class ResumeInput(BaseModel):
    resume_text: str

class KeywordOutput(BaseModel):
    keywords: str