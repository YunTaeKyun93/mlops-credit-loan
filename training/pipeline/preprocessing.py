import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
)

ONE_HOT_FEATURES = ["married", "self_employed"]
LABEL_FEATURES = ["property_area", "family_dependents"]
NUMERIC_FEATURES = ["applicant_income", "coapplicant_income", "loan_amount_term"]
REPLACE_MAP = {
    "gender": {"Male": 1, "Female": 0},
    "education": {"Graduate": 1, "Not Graduate": 0},
    "loan_status": {"Loan Default": 1, "Creditworthy": 0},
}
FEATURE_COLS = [
    "gender",
    "family_dependents",
    "education",
    "applicant_income",
    "coapplicant_income",
    "loan_amount_term",
    "credit_history",
    "property_area",
    "married_No",
    "married_Yes",
    "self_employed_No",
    "self_employed_Yes",
]
TARGET_COL = "loan_status"


def fit_preprocessors(df: pd.DataFrame) -> dict:
    """학습 데이터로 전처리기를 훈련하고 딕셔너리로 반환한다."""
    preprocessors = {}

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(df[ONE_HOT_FEATURES])
    preprocessors["one_hot_encoder"] = one_hot_encoder

    label_encoders = {}
    for col in LABEL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le
    preprocessors["label_encoders"] = label_encoders

    standard_scalers = {}
    for col in NUMERIC_FEATURES:
        ss = StandardScaler()
        ss.fit(df[[col]])
        standard_scalers[col] = ss
    preprocessors["standard_scalers"] = standard_scalers

    min_max_scalers = {}
    for col in NUMERIC_FEATURES:
        scaled = standard_scalers[col].transform(df[[col]])
        mm = MinMaxScaler()
        mm.fit(scaled)
        min_max_scalers[col] = mm
    preprocessors["min_max_scalers"] = min_max_scalers

    return preprocessors


def transform(df: pd.DataFrame, preprocessors: dict) -> tuple[pd.DataFrame, pd.Series]:
    """전처리기를 적용해 피처 행렬과 타겟을 반환한다."""
    df = df.copy()

    df["family_dependents"] = df["family_dependents"].fillna("0")
    df["loan_amount_term"] = df["loan_amount_term"].fillna(60)

    for col, mapping in REPLACE_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    ohe = preprocessors["one_hot_encoder"]
    encoded = ohe.transform(df[ONE_HOT_FEATURES]).toarray()
    ohe_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ONE_HOT_FEATURES))
    df = pd.concat([df.drop(columns=ONE_HOT_FEATURES), ohe_df], axis=1)

    for col, le in preprocessors["label_encoders"].items():
        df[col] = le.transform(df[col])

    for col, ss in preprocessors["standard_scalers"].items():
        df[col] = ss.transform(df[[col]])

    for col, mm in preprocessors["min_max_scalers"].items():
        df[col] = mm.transform(df[[col]])

    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return df[FEATURE_COLS], y


def save_preprocessors(preprocessors: dict, path: str) -> None:
    joblib.dump(preprocessors, path)


def load_preprocessors(path: str) -> dict:
    return joblib.load(path)
