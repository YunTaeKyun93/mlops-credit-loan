# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLOps pipeline for a credit loan eligibility model (`ineligible_loan_model`). The system predicts whether a loan applicant is creditworthy or a loan default risk using a Logistic Regression classifier trained on applicant data stored in MariaDB.

## Running the Stack

Copy `.env.example` to `.env` and fill in required values before starting:
```bash
cp .env.example .env
# Generate the Airflow Fernet key:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Start all services:
```bash
docker-compose up -d
```

Start a specific service (e.g., API only):
```bash
docker-compose up -d api mlflow mariadb
```

Rebuild after code changes:
```bash
docker-compose build api
docker-compose up -d api
```

## Service Endpoints

| Service | URL |
|---|---|
| API (via Nginx) | http://localhost:80 |
| API (direct) | http://localhost:8000 |
| MLflow UI | http://localhost:5001 |
| Airflow UI | http://localhost:8080 |
| MariaDB | localhost:3306 |

## Architecture

```
Nginx (80) → FastAPI API (8000)
                ├── MLflow Registry → loads Production model + preprocessors
                └── MariaDB → source data for training & batch prediction

Airflow → orchestrates training pipeline
           └── training/ pipeline modules
                ├── data_extraction.py  (MariaDB → DataFrame)
                ├── preprocessing.py    (fit/transform sklearn preprocessors)
                ├── training.py         (LogisticRegression + MLflow logging)
                └── evaluation.py       (accuracy + cumulative lift@10)
```

### Key Design Decisions

**Preprocessors are versioned with the model.** During training, fitted sklearn preprocessors (OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler) are serialized via joblib and logged as an MLflow artifact (`preprocessors/preprocessors.joblib`) in the same run as the model. The API loads them at startup from the same run as the Production model — ensuring inference preprocessing always matches training.

**`training/` is shared between Airflow and the API.** The Airflow Dockerfile sets `PYTHONPATH=/opt/airflow/training`, and the API imports `training.pipeline.preprocessing.transform` directly. This means any change to the `transform()` function affects both the Airflow training pipeline and live inference.

**Model promotion is gated by metrics.** `evaluation.is_promotable()` enforces `test_accuracy >= 0.80` and `cum_lift_10 >= 1.5` before a model version is promoted to Production stage in the MLflow registry.

### Data Flow

1. **Training**: Airflow DAG → `data_extraction.extract_features()` reads from `temp.ineligible_loan_model_features_target` (filtered by `base_ym`) → `preprocessing.fit_preprocessors()` + `transform()` → `training.train_model()` → `evaluation.evaluate()` → if `is_promotable()`, register and promote in MLflow
2. **Batch Prediction**: Airflow DAG → `data_extraction.extract_predict_targets()` reads from `mlops.ineligible_loan_model_features` (filtered by `base_dt`) → transform → predict → write results
3. **Real-time Inference**: `POST /predict` → `PreprocessService.transform()` (reuses `training.pipeline.preprocessing.transform`) → `ModelService.predict()` → returns `predict` (0/1) + `probability`

### Feature Schema

Input features: `gender`, `married`, `family_dependents`, `education`, `self_employed`, `applicant_income`, `coapplicant_income`, `loan_amount_term`, `credit_history`, `property_area`

Target: `loan_status` — `"Loan Default"` (1) or `"Creditworthy"` (0)

After preprocessing, the model receives 12 features (categorical fields are OHE/label-encoded; numeric fields are both standard- and min-max-scaled).

## MLflow Model Management

Models are loaded by name + stage. The env vars `MODEL_NAME` (default: `ineligible_loan_model`) and `MODEL_STAGE` (default: `Production`) control which model version the API serves. Changing the Production version in the MLflow registry takes effect on next API restart.

## Database

MariaDB init scripts and seed data live in `mariadb/data/docker-entrypoint-initdb.d/`. The Airflow pipeline reads from the `temp` schema (training features) and `mlops` schema (batch prediction targets).

## 개발 현황

### Phase 1 — 골격 구축 ✅ 완료
- docker-compose.yml (MariaDB + MLflow + Airflow + FastAPI + Nginx)
- api/ 구조 (routers, services, schemas, Dockerfile 멀티스테이지)
- training/pipeline/ 구조 (data_extraction.py, lift.py)
- mariadb/ 초기화 SQL

### Phase 2 — 파이프라인 구현 ✅ 완료
- [x] training/pipeline/preprocessing.py
- [x] training/pipeline/training.py
- [x] training/pipeline/evaluation.py
- [x] airflow/dags/loan_ct_dag.py
- [x] api/services/model_service.py
- [x] api/services/preprocess_service.py

### Phase 3 — 통합 테스트 ✅ 완료
- [x] docker-compose up 전체 스택 실행
- [x] 첫 MLflow 실험 로깅 확인
- [x] Airflow DAG 실행

## Python 환경
- conda mlops-study (3.11.14)
- pytest 실행: conda run -n mlops-study python -m pytest
