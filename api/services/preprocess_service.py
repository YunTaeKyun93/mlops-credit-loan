import os
import mlflow
import joblib
import numpy as np
from api.schemas.request import LoanApplicantRequest


class PreprocessService:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "ineligible_loan_model")
        self.model_stage = os.getenv("MODEL_STAGE", "Production")
        self._preprocessors: dict | None = None

    def load(self) -> None:
        """MLflow에서 학습 시 저장된 전처리 artifacts를 로드한다."""
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
        run_id = versions[0].run_id

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessors/preprocessors.joblib",
        )
        self._preprocessors = joblib.load(local_path)

    def transform(self, req: LoanApplicantRequest) -> np.ndarray:
        """요청 객체를 모델 입력 피처 배열로 변환한다."""
        from training.pipeline.preprocessing import transform
        import pandas as pd

        df = pd.DataFrame([req.model_dump()])
        features, _ = transform(df, self._preprocessors)
        return features.values
