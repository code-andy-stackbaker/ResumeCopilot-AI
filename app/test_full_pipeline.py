# import os
# import sys # Still used for sys.exit on import error
# import logging
# import torch.multiprocessing

# # If you run this script from the project root using `python -m app.test_full_pipeline`,
# # Python handles the package structure.
# # If you navigate into `app` and run `python test_full_pipeline.py`,
# # relative imports like `from recommender...` also work.

# try:
#     # Direct imports as modules are now peers or in subdirectories
#     # relative to the 'app' package context.
#     from .recommender.job_recommender import JobRecommender
#     from .langchain_qa.services import QAService
# except ImportError as e:
#     print(f"Error importing modules: {e}")
#     print("Ensure you are running this script from the project root (e.g., 'python -m app.test_full_pipeline')")
#     print("or that your PYTHONPATH is set up correctly if running from elsewhere.")
#     sys.exit(1)


# # Setup logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Recommended to include if you encounter multiprocessing issues with PyTorch
# try:
#     torch.multiprocessing.set_start_method("spawn", force=True)
# except RuntimeError:
#     # This error means it's already been set or not available on the system,
#     # which is usually fine.
#     logger.warning("torch.multiprocessing.set_start_method('spawn') has already been set or is not available.")


# def run_integrated_job_pipeline():
#     logger.info("🚀 Starting integrated job pipeline test (script inside 'app' folder)...\n")

#     # --- 1. Setup and Configuration ---
#     # Get the directory where this script (test_full_pipeline.py) is located.
#     # This will be the 'app' directory.
#     script_dir = os.path.dirname(os.path.abspath(__file__))

#     # Paths for JobRecommender model files, relative to the 'app' directory
#     faiss_recommender_index_path = os.path.join(script_dir, "recommender/model/job_index.faiss")
#     metadata_recommender_path = os.path.join(script_dir, "recommender/model/job_metadata.csv")

#     # Sample Resume
#     resume_text = (
#         "Experienced React.js developer with over 5 years of expertise in building scalable "
#         "front-end applications and user interfaces. Proficient in state management with Redux and Context API. "
#         "Strong skills in Node.js and Express.js for backend development, creating RESTful APIs. "
#         "Familiar with AWS services like S3, EC2, and Lambda for deployment and cloud solutions. "
#         "Proven ability in database management with MongoDB and PostgreSQL. "
#         "A collaborative team player with excellent problem-solving skills and a passion for learning new technologies."
#     )
#     logger.info(f"📄 Sample Resume:\n{resume_text}\n")

#     # --- 2. Initialize Services ---
#     try:
#         logger.info("Initializing JobRecommender...")
#         recommender = JobRecommender(
#             faiss_index_path=faiss_recommender_index_path,
#             metadata_path=metadata_recommender_path,
#             top_k=3
#         )
#         logger.info("JobRecommender initialized successfully.")

#         logger.info("Initializing QAService...")
#         # QAService path configurations are internal or absolute, so no changes needed here for it.
#         qa_service = QAService()
#         logger.info("QAService initialized successfully.")

#     except FileNotFoundError as e:
#         logger.error(f"❌ ERROR: Critical file not found during service initialization. Please check paths: {e}")
#         logger.error(f"Attempted Recommender FAISS index path: {faiss_recommender_index_path}")
#         logger.error(f"Attempted Recommender metadata path: {metadata_recommender_path}")
#         logger.error("Ensure all model and index files are correctly placed and paths are accurate.")
#         return
#     except Exception as e:
#         logger.error(f"❌ ERROR during service initialization: {e}", exc_info=True)
#         return

#     # --- 3. Job Recommendation & Classification (Handled by JobRecommender) ---
#     logger.info("🔍 Phase 1: Getting Job Recommendations (Recommender + In-built Classifier)...")
#     try:
#         recommendations = recommender.recommend(resume_text)
#     except Exception as e:
#         logger.error(f"❌ ERROR during recommendation phase: {e}", exc_info=True)
#         return

#     if not recommendations:
#         logger.warning("🤷 No recommendations found for the given resume.")
#         return

#     logger.info("\n🏆 Top Recommended Jobs (after FAISS + Classifier reranking by JobRecommender):")
#     for i, job in enumerate(recommendations):
#         # Using .get with a default for robustness, in case a score is missing
#         faiss_score_str = f"{job.get('faiss_score', 'N/A'):.4f}" if isinstance(job.get('faiss_score'), float) else str(job.get('faiss_score', 'N/A'))
#         classifier_score_str = f"{job.get('classifier_score', 'N/A'):.4f}" if isinstance(job.get('classifier_score'), float) else str(job.get('classifier_score', 'N/A'))
        
#         logger.info(f"  {i+1}. Job Description (Snippet): {job.get('job_description', 'N/A')[:150]}...")
#         logger.info(f"     FAISS Score: {faiss_score_str}, Classifier Score: {classifier_score_str}")


#     # --- 4. Specific Q&A about the Top Recommended Job using QAService ---
#     logger.info("\n💬 Phase 2: Detailed Q&A about the Top Recommended Job...")
    
#     if not recommendations: # Should have returned earlier, but as a safeguard
#         logger.warning("No recommendations to select for Q&A.")
#         return

#     selected_job = recommendations[0]
#     selected_job_description = selected_job.get('job_description', 'No description available.')
#     classifier_score_for_selected_job = selected_job.get('classifier_score', 'N/A')
#     classifier_score_str_selected = f"{classifier_score_for_selected_job:.4f}" if isinstance(classifier_score_for_selected_job, float) else str(classifier_score_for_selected_job)


#     logger.info(f"\nSelected Job for Q&A (Classifier Score: {classifier_score_str_selected}):")
#     logger.info(f"{selected_job_description[:300]}...\n")

#     question_about_match = (
#       "What skills required for this selected job? "
#     )
#     logger.info(f"❓ Question for LLM: {question_about_match}\n")

#     context_for_llm = f"Candidate's Resume:\n{resume_text}\n\nSpecific Job Description for Analysis:\n{selected_job_description}"
#     general_question = "what are required skills in this job?"
    
#     try:
#         logger.info("🧠 Invoking QAService for specific context Q&A...")
        
#         qa_result = qa_service.answer_general_question(question_about_match)
        
#         if qa_result:
#             logger.info("\n✅ LLM Answer about Job Match:")
#             # Print directly to console for potentially better formatting of LLM output
#             print(qa_result)
#         elif qa_result and "error" in qa_result:
#             logger.error(f"❌ Q&A Service returned an error: {qa_result}")
#         else:
#             logger.error("❌ Q&A Service returned an unexpected result or no answer.")

#     except Exception as e:
#         logger.error(f"❌ ERROR during specific Langchain Q&A phase: {e}", exc_info=True)

#     logger.info("\n🏁 Integrated job pipeline test finished.")

# if __name__ == "__main__":
#     run_integrated_job_pipeline()
