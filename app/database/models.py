# app/database/models.py
from sqlalchemy import Column, Integer, String, Text, func # func is for server_default SQL functions
from sqlalchemy.dialects.postgresql import TIMESTAMP, TEXT # Using TEXT explicitly for potentially long descriptions
from pgvector.sqlalchemy import Vector # Import the Vector type for pgvector

from .session import Base # Import Base from the session.py in the same directory
from app.config import EMBEDDING_DIMENSION_RECOMMENDER # Import the embedding dimension from config

class Job(Base):
    __tablename__ = "jobs" # This will be the name of the table created in PostgreSQL

    # Define the columns for the 'jobs' table
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # It's good practice to have an external ID if your jobs come from various sources
    # or if you need a persistent identifier other than the auto-incrementing 'id'.
    # Make it unique if it should be.
    external_job_id = Column(String(255), unique=True, index=True, nullable=True) 

    title = Column(String(500), nullable=False, index=True) # Indexing for faster searches on title
    company = Column(String(255), nullable=True, index=True) # Indexing company
    location = Column(String(255), nullable=True, index=True) # Indexing location

    # Using TEXT for job descriptions as they can be quite long
    job_description_text = Column(TEXT, nullable=False) 

    # This is our vector column for storing the job description embeddings.
    # The dimension (EMBEDDING_DIMENSION_RECOMMENDER) MUST match the output
    # dimension of the sentence transformer model you are using.
    job_description_embedding = Column(Vector(EMBEDDING_DIMENSION_RECOMMENDER), nullable=True)