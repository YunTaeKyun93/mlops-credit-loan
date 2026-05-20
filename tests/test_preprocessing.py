import pandas as pd
import pytest

from training.pipeline.preprocessing import (
    FEATURE_COLS,
    fit_preprocessors,
    transform,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "married": ["Yes", "No", "Yes"],
        "family_dependents": ["0", "1", "2"],
        "education": ["Graduate", "Not Graduate", "Graduate"],
        "self_employed": ["No", "Yes", "No"],
        "applicant_income": [5000, 3000, 4000],
        "coapplicant_income": [0, 1500, 200],
        "loan_amount_term": [360, 180, 360],
        "credit_history": [1.0, 0.0, 1.0],
        "property_area": ["Urban", "Rural", "Semiurban"],
        "loan_status": ["Creditworthy", "Loan Default", "Creditworthy"],
    })


# ── fit_preprocessors ─────────────────────────────────────────────────────────

def test_fit_preprocessors_returns_all_keys(sample_df):
    preprocessors = fit_preprocessors(sample_df)
    assert set(preprocessors.keys()) == {
        "one_hot_encoder",
        "label_encoders",
        "standard_scalers",
        "min_max_scalers",
    }


def test_fit_preprocessors_ohe_feature_names(sample_df):
    ohe = fit_preprocessors(sample_df)["one_hot_encoder"]
    expected = ["married_No", "married_Yes", "self_employed_No", "self_employed_Yes"]
    assert list(ohe.get_feature_names_out(["married", "self_employed"])) == expected


def test_fit_preprocessors_label_encoders_cover_all_label_features(sample_df):
    label_encoders = fit_preprocessors(sample_df)["label_encoders"]
    assert "property_area" in label_encoders
    assert "family_dependents" in label_encoders


def test_fit_preprocessors_scalers_cover_all_numeric_features(sample_df):
    preprocessors = fit_preprocessors(sample_df)
    numeric_cols = ["applicant_income", "coapplicant_income", "loan_amount_term"]
    assert set(preprocessors["standard_scalers"].keys()) == set(numeric_cols)
    assert set(preprocessors["min_max_scalers"].keys()) == set(numeric_cols)


# ── transform ─────────────────────────────────────────────────────────────────

def test_transform_output_shape(sample_df):
    preprocessors = fit_preprocessors(sample_df)
    X, y = transform(sample_df, preprocessors)
    assert X.shape == (len(sample_df), len(FEATURE_COLS))


def test_transform_column_order(sample_df):
    preprocessors = fit_preprocessors(sample_df)
    X, _ = transform(sample_df, preprocessors)
    assert list(X.columns) == FEATURE_COLS


def test_transform_target_is_binary(sample_df):
    preprocessors = fit_preprocessors(sample_df)
    _, y = transform(sample_df, preprocessors)
    assert set(y.unique()).issubset({0, 1})


def test_transform_no_nulls_in_output(sample_df):
    preprocessors = fit_preprocessors(sample_df)
    X, _ = transform(sample_df, preprocessors)
    assert X.isnull().sum().sum() == 0


# ── fillna 처리 ───────────────────────────────────────────────────────────────

def test_transform_fillna_family_dependents(sample_df):
    """family_dependents의 결측값은 "0"으로 대체되어야 한다."""
    df = sample_df.copy()
    df.loc[0, "family_dependents"] = None
    preprocessors = fit_preprocessors(sample_df)
    X, _ = transform(df, preprocessors)
    assert X.isnull().sum().sum() == 0


def test_transform_fillna_loan_amount_term(sample_df):
    """loan_amount_term의 결측값은 60으로 대체되어야 한다."""
    df = sample_df.copy()
    df.loc[1, "loan_amount_term"] = None
    preprocessors = fit_preprocessors(sample_df)
    X, _ = transform(df, preprocessors)
    assert X.isnull().sum().sum() == 0
