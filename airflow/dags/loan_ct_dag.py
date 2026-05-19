import logging
import os
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

from training.pipeline.data_extraction import extract_features
from training.pipeline.evaluation import evaluate, is_promotable
from training.pipeline.preprocessing import fit_preprocessors, save_preprocessors, transform
from training.pipeline.training import register_model, train_model

log = logging.getLogger(__name__)

DB_URL = os.getenv("MARIADB_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "ineligible_loan_model")


@dag(
    dag_id="loan_ct_dag",
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["loan", "training"],
)
def loan_ct_dag():

    @task
    def extract(ds=None) -> str:
        cutoff_ym = ds.replace("-", "")[:6]
        df = extract_features(DB_URL, cutoff_ym)
        base_dir = f"/tmp/loan_ct_{ds.replace('-', '')}"
        os.makedirs(base_dir, exist_ok=True)
        df.to_parquet(f"{base_dir}/raw.parquet", index=False)
        log.info("Extracted %d rows (cutoff_ym=%s)", len(df), cutoff_ym)
        return base_dir

    @task
    def preprocess(base_dir: str) -> str:
        df = pd.read_parquet(f"{base_dir}/raw.parquet")
        preprocessors = fit_preprocessors(df)
        save_preprocessors(preprocessors, f"{base_dir}/preprocessors.joblib")

        x, y = transform(df, preprocessors)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        x_train.to_parquet(f"{base_dir}/x_train.parquet", index=False)
        x_test.to_parquet(f"{base_dir}/x_test.parquet", index=False)
        y_train.to_frame().to_parquet(f"{base_dir}/y_train.parquet", index=False)
        y_test.to_frame().to_parquet(f"{base_dir}/y_test.parquet", index=False)
        log.info("Preprocessed: train=%d, test=%d", len(x_train), len(x_test))
        return base_dir

    @task
    def train(base_dir: str) -> str:
        x_train = pd.read_parquet(f"{base_dir}/x_train.parquet")
        y_train = pd.read_parquet(f"{base_dir}/y_train.parquet").squeeze()
        with mlflow.start_run() as run:
            model = train_model(x_train, y_train)
            run_id = run.info.run_id
        joblib.dump(model, f"{base_dir}/model.joblib")
        log.info("Trained model (run_id=%s)", run_id)
        return run_id

    @task
    def evaluate_model(base_dir: str, mlflow_run_id: str) -> bool:
        model = joblib.load(f"{base_dir}/model.joblib")
        x_train = pd.read_parquet(f"{base_dir}/x_train.parquet")
        y_train = pd.read_parquet(f"{base_dir}/y_train.parquet").squeeze()
        x_test = pd.read_parquet(f"{base_dir}/x_test.parquet")
        y_test = pd.read_parquet(f"{base_dir}/y_test.parquet").squeeze()
        with mlflow.start_run(run_id=mlflow_run_id):
            metrics = evaluate(model, x_train, y_train, x_test, y_test)
        promotable = is_promotable(metrics)
        log.info("Metrics: %s | promotable=%s", metrics, promotable)
        return promotable

    @task
    def register(base_dir: str, promotable: bool, mlflow_run_id: str) -> None:
        if not promotable:
            log.info(
                "Skipping registration: promotion criteria not met "
                "(min_accuracy=0.80, min_cum_lift_10=1.5)"
            )
            raise AirflowSkipException("Promotion criteria not met")
        model = joblib.load(f"{base_dir}/model.joblib")
        with mlflow.start_run(run_id=mlflow_run_id):
            register_model(model, MODEL_NAME, f"{base_dir}/preprocessors.joblib")
        client = mlflow.MlflowClient()
        version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info("Promoted version %s to Production", version)

    base_dir = extract()
    base_dir = preprocess(base_dir)
    mlflow_run_id = train(base_dir)
    promotable = evaluate_model(base_dir, mlflow_run_id)
    register(base_dir, promotable, mlflow_run_id)


loan_ct_dag()
