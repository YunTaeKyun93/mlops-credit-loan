import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 100,
    max_iter: int = 100,
) -> LogisticRegression:
    """로지스틱 회귀 모델을 학습하고 MLflow에 파라미터를 로깅한다."""
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "random_state": random_state,
        "max_iter": max_iter,
        "train_size": len(x_train),
    })

    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def register_model(
    model: LogisticRegression,
    model_name: str,
    preprocessors_path: str,
) -> str:
    """모델과 전처리 artifacts를 MLflow에 로깅하고 레지스트리에 등록한다."""
    mlflow.log_artifact(preprocessors_path, artifact_path="preprocessors")
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
    )
    return model_info.model_uri
