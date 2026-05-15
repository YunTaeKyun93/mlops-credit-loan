import os
import mlflow.sklearn


class ModelService:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "ineligible_loan_model")
        self.model_stage = os.getenv("MODEL_STAGE", "Production")
        self.model_version: str | None = None
        self._model = None

    def load(self) -> None:
        """MLflow Model Registry에서 Production 모델을 로드한다."""
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        self._model = mlflow.sklearn.load_model(model_uri)

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
        self.model_version = versions[0].version if versions else "unknown"

    def predict(self, features) -> dict:
        proba = self._model.predict_proba(features)[0]
        predict = int(self._model.predict(features)[0])
        return {"predict": predict, "probability": round(float(proba[1]), 6)}
