FROM python:3.11-slim

WORKDIR /mlflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
