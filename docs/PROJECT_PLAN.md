# Project Plan: MLOps for Job Recommendation & Q/A RAG System (7-Day Sprint)

## Phase 1: Local Development & MLOps Foundation (2 Days)

This phase focuses on completing and testing your core automation and application packaging locally before moving to the cloud.

### Day 1: Design Nightly Embedding Refresh DAG (Local Airflow)

**Daily Objective:** Create a local Airflow DAG to automate the process of fetching new data, generating embeddings, and updating pgvector.
**Main Tasks:**

- Develop an Airflow DAG that simulates fetching new job/resume data.
- Integrate your sentence transformer script to generate embeddings.
- Write tasks to connect to your local PostgreSQL (with pgvector) and update/insert new embeddings.
- Test the DAG locally to ensure it runs end-to-end.
  **Key Tools:** Apache Airflow (local), Python, sentence-transformers, psycopg2 (or other PostgreSQL connector), PostgreSQL (with pgvector).
  **MLOps Note:** Focus on modular DAG design and clear logging. This DAG will be migrated to AWS later.

### Day 2: Containerize & Test Full Application Stack (Local Docker)

**Daily Objective:** Package your FastAPI application and PostgreSQL (with pgvector) into Docker containers and test the entire system locally.
**Main Tasks:**

- Create a Dockerfile for your FastAPI application, ensuring all dependencies are included and it can connect to PostgreSQL.
- Set up `docker-compose.yml` to run:
  - Your FastAPI application container.
  - A PostgreSQL container with the pgvector extension enabled.
- Launch the full stack using `docker-compose` and perform end-to-end tests on relevant API endpoints (e.g., `/recommend_jobs`, `/job_qa`) against the containerized pgvector. (Note: The `/classify_match` endpoint's functionality will rely on similarity scores from pgvector or simple heuristics if it remains, as no separate classifier is being trained/deployed in this plan).
  **Key Tools:** Docker, docker-compose, Gunicorn.
  **MLOps Note:** Aim for optimized and lean Docker images (e.g., using multi-stage builds). This ensures consistency between local development and cloud deployment.

## Phase 2: Cloud Deployment & MLOps Automation on AWS (5 Days)

This phase focuses on migrating your application and the embedding MLOps pipeline to AWS, making them production-ready.

### Day 3: Establish AWS Foundations for MLOps

**Daily Objective:** Set up the core AWS infrastructure and services needed for your deployment.
**Main Tasks:**

- Configure IAM roles and users with the principle of least privilege for all services (Airflow, ECS, S3, RDS, potentially SageMaker for processing jobs).
- Create and configure S3 buckets (with versioning) for datasets and logs. (S3 for model artifacts for a custom classifier is no longer a primary concern).
- Provision an Amazon RDS for PostgreSQL instance, ensuring the pgvector extension is installed and security groups are correctly configured for access.
- Set up your SageMaker Studio environment (still useful for data exploration, script development, or managing SageMaker Processing jobs if used for embeddings).
- (Highly Recommended): Begin setting up a managed Airflow environment like AWS MWAA to simplify operations.
  **Key Tools:** AWS Console, AWS CLI, IAM, S3, Amazon RDS for PostgreSQL, SageMaker Studio, (Optional: AWS MWAA).
  **MLOps Note:** Document your infrastructure setup. Ideally, use Infrastructure as Code (IaC) tools like Terraform or AWS CDK in the future, but for this sprint, clear documentation is key if manual.

### Day 4: Deploy FastAPI Application to AWS ECS

**Daily Objective:** Get your FastAPI application running on AWS and accessible via API Gateway.
**Main Tasks:**

- Push your FastAPI Docker image (from Day 2) to Amazon ECR (Elastic Container Registry).
- Set up Amazon ECS:
  - Create an ECS Cluster.
  - Define an ECS Task Definition (using Fargate for serverless compute) for your FastAPI app, specifying resource needs, image URI from ECR, environment variables (e.g., for RDS connection), and logging.
  - Create an ECS Service to run and maintain your tasks.
- Configure an Application Load Balancer (ALB) to route traffic to your ECS service.
- Set up API Gateway to expose your FastAPI endpoints publicly, connecting it to the ALB.
  **Key Tools:** Amazon ECR, Amazon ECS (Fargate), Application Load Balancer, API Gateway.
  **MLOps Note:** This setup provides scalability and resilience for your API. Ensure health checks are properly configured in ECS and ALB.

### Day 5: Automate pgvector Updates in the Cloud (Airflow on AWS)

**Daily Objective:** Migrate and test your pgvector embedding refresh pipeline on AWS using Airflow.
**Main Tasks:**

- Configure your cloud-based Airflow (MWAA or EC2-based) with secure connections to RDS (pgvector) and S3 (for job data sources).
- Deploy your "Nightly Embedding Refresh" DAG (from Day 1) to this Airflow environment.
- Determine the execution environment for the embedding script itself:
  - Recommended: As a SageMaker Processing Job (for good resource management and MLOps integration) or a dedicated ECS Task, triggered by Airflow.
- Conduct an end-to-end test to ensure the deployed DAG correctly updates pgvector in RDS.
  **Key Tools:** Apache Airflow (ideally MWAA), S3, Amazon RDS for PostgreSQL, IAM, (Optional/Recommended: SageMaker Processing or ECS).
  **MLOps Note:** Use AWS Secrets Manager for handling database credentials and other secrets in Airflow. Monitor DAG execution closely.

### Day 6: Implement CI/CD and Basic Monitoring

**Daily Objective:** Establish basic CI/CD for your API and set up initial monitoring for your deployed services.
**Main Tasks:**

- CI/CD for FastAPI: Create a simple CI/CD pipeline (e.g., using GitHub Actions or AWS CodePipeline). On a push to your main branch, it should:
  - Build the FastAPI Docker image.
  - Push the image to Amazon ECR.
  - Update the ECS service to deploy the new version.
- Monitoring & Logging:
  - Set up Amazon CloudWatch Logs for your FastAPI application running on ECS.
  - Create basic CloudWatch Alarms for key metrics of your ECS service (CPU/memory usage, error rates from ALB) and any critical SageMaker Processing job metrics (if SageMaker is used for embeddings).
  - Review Airflow logs for DAG health.
- Review and refine all system configurations for robustness and security.
  **Key Tools:** GitHub Actions (or AWS CodePipeline), Amazon CloudWatch (Logs, Metrics, Alarms), ECR, ECS.
  **MLOps Note:** This automates application deployment and provides initial observability.

### Day 7: Thorough End-to-End Testing, Documentation & Demo

**Daily Objective:** Ensure the entire system works as expected, document it thoroughly, and prepare a demonstration.
**Main Tasks:**

- Perform comprehensive end-to-end testing of all components in the AWS environment: API functionality, Airflow DAG executions (embedding refresh), and data consistency.
- Finalize your project `README.md` with:
  - A clear architecture diagram of the cloud deployment.
  - Detailed setup and deployment instructions.
  - API usage examples.
  - An overview of the MLOps pipelines (now focused on embedding refresh).
- Prepare and record a demonstration of the live, functioning system, highlighting its key features and the automated embedding pipeline.
  **Key Tools:** Postman/curl, AWS Management Console, Markdown, OBS Studio (or other recording software).
  **MLOps Note:** Good documentation is crucial for maintainability, collaboration, and future development. The demo should showcase the MLOps maturity achieved for the deployed system.
