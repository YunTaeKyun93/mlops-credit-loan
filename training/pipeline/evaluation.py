import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from training.support.model.evaluation.lift import Lift


def evaluate(
    model: LogisticRegression,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """모델 성능을 평가하고 MLflow에 metrics를 로깅한다.

    Returns:
        {"train_accuracy": float, "test_accuracy": float, "cum_lift_10": float}
    """
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)

    test_proba = pd.DataFrame(model.predict_proba(x_test))[1].to_list()
    lift = Lift(probabilities=test_proba, labels=y_test.to_list(), cut_count=10)
    cum_lift_10 = lift.get_first_cum_lift()

    metrics = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "cum_lift_10": round(cum_lift_10, 4),
    }
    mlflow.log_metrics(metrics)
    return metrics


def is_promotable(metrics: dict, min_accuracy: float = 0.80, min_lift: float = 1.5) -> bool:
    """Production 승격 기준을 충족하는지 검사한다."""
    return (
        metrics["test_accuracy"] >= min_accuracy
        and metrics["cum_lift_10"] >= min_lift
    )
