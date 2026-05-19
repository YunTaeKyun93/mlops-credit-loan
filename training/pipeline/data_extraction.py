import pandas as pd
from sqlalchemy import create_engine, text


def extract_features(db_url: str, cutoff_ym: str) -> pd.DataFrame:
    """MariaDB에서 학습용 피처+타겟 데이터를 추출한다.

    Args:
        db_url: SQLAlchemy 연결 문자열
        cutoff_ym: 학습 기준 연월 (YYYYMM)

    Returns:
        ineligible_loan_model_features_target 테이블 DataFrame
    """
    sql = """
        SELECT *
          FROM mlops.ineligible_loan_model_features_target
         WHERE base_ym <= :cutoff_ym
    """
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"cutoff_ym": cutoff_ym})
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def extract_predict_targets(db_url: str, base_dt: str) -> pd.DataFrame:
    """배치 예측 대상 신청자의 피처 데이터를 추출한다.

    Args:
        db_url: SQLAlchemy 연결 문자열
        base_dt: 기준일자 (YYYYMMDD)

    Returns:
        ineligible_loan_model_features 테이블 DataFrame
    """
    sql = """
        SELECT f.*
          FROM mlops.ineligible_loan_model_features f
         WHERE f.base_dt = :base_dt
    """
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(sql), {"base_dt": base_dt})
        return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
