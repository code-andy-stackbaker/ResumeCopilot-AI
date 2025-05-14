# app/database/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DATABASE_URL  # Import the DATABASE_URL from your app's config

# 1. Create the SQLAlchemy engine
# The engine is the entry point to the database and manages the DBAPI connection pool.
# echo=False by default. Set echo=True if you want to see all SQL statements SQLAlchemy generates.
engine = create_engine(DATABASE_URL)

# 2. Create a SessionLocal class
# This class will be a factory for new Session objects (database sessions).
# - autocommit=False: Changes are not automatically committed to the database.
#                    You need to call session.commit() explicitly.
# - autoflush=False: Changes are not automatically sent to the DB upon every modification
#                   within a session, but rather when data is queried or explicitly flushed.
# - bind=engine: Associates this session factory with our database engine.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. Create a Base class for declarative model definitions
# All your ORM models (classes representing tables) will inherit from this Base.
# It keeps a catalog of classes and tables related to that base.
Base = declarative_base()


# --- Optional: Dependency for FastAPI or other frameworks ---
# This function can be used as a dependency (e.g., in FastAPI routes)
# to provide a database session for the duration of a request and ensure it's closed.
def get_db_session():
    db = SessionLocal()
    try:
        yield db  # Provides the session to the code that needs it
    finally:
        db.close() # Ensures the session is closed properly after use

if __name__ == "__main__":
    # This block is for testing purposes if you run this file directly
    # e.g., from your project root: python -m app.database.session
    print(f"Attempting to connect to database using URL: {DATABASE_URL}")
    try:
        # Test the connection by creating a connection from the engine
        with engine.connect() as connection:
            print("SQLAlchemy engine successfully connected to the database!")
            print("Next: Define your ORM models in 'app/database/models.py' inheriting from 'Base'.")
    except Exception as e:
        print(f"Failed to connect to the database using the SQLAlchemy engine.")
        print(f"Error: {e}")
        print("Please ensure your PostgreSQL Docker container ('local_job_db_pg') is running,")
        print("and your '.env' file and 'app/config.py' are correctly configured with database details.")