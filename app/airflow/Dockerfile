FROM apache/airflow:2.10.2
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary \
    transformers==4.45.0 \
    torch==2.3.1 \
    peft \
    accelerate \
    bitsandbytes \
    qwen-vl-utils \
    pillow \
    scikit-learn \
    pandas \
    sqlalchemy \
    numpy