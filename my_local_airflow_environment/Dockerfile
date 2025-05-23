# LLM_DeepLearning/my_local_airflow_environment/Dockerfile
FROM apache/airflow:2.9.2-python3.11
ARG CACHE_BUSTER_ARG=1
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgmp-dev \
        libmpfr-dev \
        libmpc-dev && \
    rm -rf /var/lib/apt/lists/*
COPY my_local_airflow_environment/requirements-airflow.txt /requirements.txt 
RUN chmod +r /requirements.txt
USER airflow
RUN echo "--- [DOCKER_BUILD_DEBUG] Attempting pip install from /requirements.txt (Airflow specific) ---" && \
    echo "Python version during pip install: $(python --version)" && \
    echo "Pip version during pip install: $(pip --version)" && \
    pip install --verbose --no-cache-dir -r /requirements.txt