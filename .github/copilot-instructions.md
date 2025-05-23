# Project Overview: MLOps for Job Recommendation & Q/A RAG System

This project focuses on building and deploying an MLOps system for a job recommendation and Q/A RAG (Retrieval Augmented Generation) service. The core is an automated pipeline for refreshing vector embeddings in pgvector. Development includes local setup and Dockerization (Phase 1), followed by cloud deployment on AWS with MLOps automation (Phase 2). The full detailed plan is in `PROJECT_PLAN.md` (or `docs/PROJECT_PLAN.md`).

## Core Technologies & Stack:

**Local Development (Phase 1):**

- **Programming Language:** Python
- **Web Framework:** FastAPI
- **Database & Embeddings:** PostgreSQL with pgvector extension
- **Embedding Generation:** sentence-transformers library (likely using models like Sentence-BERT, which rely on Hugging Face Transformers and PyTorch/TensorFlow)
- **Orchestration:** Apache Airflow (local instance)
- **Containerization:** Docker, docker-compose
- **Database Interaction:** psycopg2 (or other relevant Python PostgreSQL connector like SQLAlchemy if used)

**Cloud Deployment (AWS - Phase 2):**

- **Compute for API:** Amazon ECS (Fargate)
- **Container Registry:** Amazon ECR
- **API Exposure:** Amazon API Gateway, Application Load Balancer (ALB)
- **Database:** Amazon RDS for PostgreSQL (with pgvector extension)
- **Storage (Data, Logs):** Amazon S3 (with versioning)
- **ML Processing (for embeddings, if not run on ECS/Airflow worker):** AWS SageMaker (Studio for development/exploration, SageMaker Processing Jobs for script execution)
- **Orchestration:** AWS Managed Workflows for Apache Airflow (MWAA) or self-managed Airflow on EC2
- **CI/CD:** GitHub Actions or AWS CodePipeline
- **Monitoring & Logging:** Amazon CloudWatch (Logs, Metrics, Alarms)
- **Identity & Access:** AWS IAM (roles, users, least privilege)
- **Secrets Management:** AWS Secrets Manager

## Key MLOps Principles & Guidelines (from PROJECT_PLAN.md):

- **Idempotency:** Ensure Airflow DAGs for embedding refresh can be rerun without unintended side effects.
- **Parameterization:** Design DAGs and scripts with configurable parameters.
- **Versioning:** Implement version control for code, data sources (if applicable), and Docker images.
- **Containerization:** Utilize Docker for consistent environments from local development to cloud deployment. Aim for lean, optimized images.
- **Automation:** Automate the embedding refresh workflow using Airflow and CI/CD pipelines for the application.
- **Infrastructure as Code (IaC):** Document manual setups now; aim for IaC (Terraform/CDK) in the future.
- **Modularity & Reusability:** Design components (scripts, DAG tasks) to be modular.
- **Logging & Monitoring:** Implement comprehensive logging in Airflow DAGs, the FastAPI app, and any embedding generation jobs. Set up basic CloudWatch monitoring.
- **Security:** Adhere to the principle of least privilege for IAM. Use AWS Secrets Manager for credentials. Configure security groups.
- **Documentation:** Maintain clear and thorough documentation (especially `README.md` with architecture, setup, API usage, MLOps overview for the embedding pipeline).
- **Testing:** Emphasize local testing of DAGs and containerized applications. Conduct thorough end-to-end testing in the cloud.

## Development Phases Overview (from PROJECT_PLAN.md):

- **Phase 1 (Local - 2 Days):** Design and test the local Airflow DAG for pgvector embedding refresh. Containerize the FastAPI application with PostgreSQL/pgvector using Docker.
- **Phase 2 (Cloud - AWS - 5 Days):** Deploy the FastAPI application and the automated embedding refresh pipeline to AWS. Implement CI/CD for the API and set up monitoring.

## General Coding Guidelines:

- Python code should be well-commented and generally follow PEP 8 guidelines.
- Airflow DAGs should clearly define tasks, dependencies, and include error handling and robust logging.
- FastAPI endpoints should be well-defined.
- When interacting with AWS services, use the AWS SDK (Boto3 for Python) following best practices.
- For any specific daily tasks, refer to the detailed `docs/PROJECT_PLAN.md`.

## General copilor prompts for every project

- Always explain every step, even the basics, and provide beginner-level, step-by-step guides.
- Show all commands, file edits, and configuration changes in full, even if they seem obvious.
- Do not assume prior knowledge—break down each action, command, and code edit in detail.
- List and explain all prerequisites for any task or implementation.
- If there are multiple ways to do something, mention the most common or recommended one.
- If troubleshooting is likely, include common errors and how to fix them.
- When I ask for an implementation, fetch and use the maximum context from the project (including all related files, configurations, and documentation).
- Provide complete code for the requested file, including all imports, exports, and comments.
- If the functionality depends on other files or modules, also provide their full implementation (not just stubs).
- Show the full content of all files involved—not just code snippets.
- Clearly explain how files and components connect and interact within the project.
- Clearly list any assumptions, required configurations, or environment setups.
- If the implementation is long, separate each file with its filename and language as a heading.
- Explain cryptic error messages in plain English.
- Provide relevant background context for any specific technologies involved (e.g., Docker, Node.js, React, FastAPI, nest.js, Next.js, Python, AI, LLM, Pytorch).

**:Example usage:**

- "When I ask for a feature or file, or solution to any error, give me the full code for that file and any other files it depends on, with explanations and all necessary context."
