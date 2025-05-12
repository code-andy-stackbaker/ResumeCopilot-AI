from pydantic import BaseModel,  field_validator

class ResumeInput(BaseModel):
    resume_text: str

class KeywordOutput(BaseModel):
    keywords: str
    
class ResumeInput(BaseModel):
  resume_text: str

@field_validator('resume_text')
@classmethod
def resume_text_not_empty(cls, value):
  if not value.strip():
      raise ValueError('Resume text cannot be empty or contain only whitespace.')
  return value    