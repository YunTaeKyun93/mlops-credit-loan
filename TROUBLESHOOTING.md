# Phase 3 트러블슈팅 기록

---

## 1. pandas `read_sql`에 SQLAlchemy Engine 직접 전달 오류

**에러 메시지**
```
AttributeError: 'Engine' object has no attribute 'cursor'
```

**원인**
pandas 2.x에서 `pd.read_sql(sql, con=engine)`으로 Engine 객체를 직접 넘기면, 내부적으로 DBAPI2 커넥션으로 오인해 `.cursor()`를 호출하려다 실패한다.

**해결 방법**
`pd.read_sql` 대신 SQLAlchemy의 `conn.execute(text(sql), params)`를 사용하고, 결과를 `pd.DataFrame(result.fetchall(), columns=list(result.keys()))`로 변환한다.

```python
with engine.connect() as conn:
    result = conn.execute(text(sql), {"cutoff_ym": cutoff_ym})
    return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
```

**핵심 교훈**
pandas 2.x와 SQLAlchemy 2.x를 함께 쓸 때는 `pd.read_sql`보다 `conn.execute()` 조합이 안전하다.

---

## 2. LabelEncoder fit/transform 불일치 — unseen label `'0'`

**에러 메시지**
```
ValueError: y contains previously unseen labels: '0'
```

**원인**
`fit_preprocessors()`는 원본 데이터 그대로 LabelEncoder를 fit하지만, `transform()`에서는 `family_dependents`의 null을 `"0"`으로 채운다. fit 시점에 `"0"`이 존재하지 않으면 transform에서 unknown label 오류가 발생한다.

**해결 방법**
`fit_preprocessors()` 내부에 `transform()`과 동일한 null 처리를 먼저 적용한다.

```python
def fit_preprocessors(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["family_dependents"] = df["family_dependents"].fillna("0")
    df["loan_amount_term"] = df["loan_amount_term"].fillna(60)
    ...
```

**핵심 교훈**
fit과 transform에 적용되는 전처리 로직은 반드시 동일해야 한다. null 처리, 인코딩 순서 등 모든 변환 단계를 fit 단계에서도 동일하게 적용할 것.

---

## 3. Airflow 예약어 충돌 — `run_id` 파라미터명

**에러 메시지**
```
ValueError: The key 'run_id' in args is a part of kwargs and therefore reserved.
```

**원인**
Airflow는 `run_id`를 DAG run 컨텍스트 변수로 예약하고 있다. `@task` 데코레이터 함수의 파라미터 이름이 `run_id`이면 Airflow가 XCom 값 대신 DAG run ID를 주입하려 해 충돌한다.

**해결 방법**
MLflow run ID를 전달하는 파라미터명을 `mlflow_run_id`로 변경한다.

```python
@task
def evaluate_model(base_dir: str, mlflow_run_id: str) -> bool:
    with mlflow.start_run(run_id=mlflow_run_id):
        ...
```

**핵심 교훈**
Airflow `@task` 파라미터명으로 `run_id`, `dag_id`, `task_id`, `ds`, `ts` 등 Airflow 컨텍스트 예약어를 사용하면 안 된다.

---

## 4. MLflow artifact 쓰기 권한 오류

**에러 메시지**
```
PermissionError: [Errno 13] Permission denied: '/mlflow'
```

**원인**
Airflow 컨테이너가 MLflow artifact를 로컬 경로(`/mlflow/artifacts`)에 직접 쓰려 하지만, 해당 경로가 컨테이너 내에 존재하지 않거나 권한이 없었다.

**해결 방법**
두 가지를 함께 적용했다.

1. MLflow 서버에 `--serve-artifacts` 플래그 추가 (HTTP를 통한 artifact 업로드 프록시 활성화)
2. `docker-compose.yml`의 Airflow 공통 볼륨에 `mlflow-artifacts` 볼륨 마운트 추가
3. 볼륨 디렉토리에 권한 부여

```bash
docker exec -u root mlops-credit-loan-airflow-scheduler-1 chmod -R 777 /mlflow/artifacts
```

**핵심 교훈**
Airflow와 MLflow가 별도 컨테이너에 있을 때는 artifact 저장소를 공유 볼륨으로 마운트하거나, MLflow에 `--serve-artifacts`를 활성화해야 한다.

---

## 5. 학습 데이터 테이블 비어있음 — `ineligible_loan_model_features_target`

**에러 메시지**
```
(데이터 0건 추출, 모델 학습 불가)
```

**원인**
`mlops.ineligible_loan_model_features_target` 테이블이 스키마만 생성되고 데이터가 없었다. 원본 피처 마트 생성 SQL(`01_data_extract.sql`)이 MariaDB init 스크립트에 포함되지 않았고, `temp` 스키마에 생성하도록 되어 있어 스키마 불일치도 존재했다.

**해결 방법**
`03_create_features.sql`을 작성해 `docker-entrypoint-initdb.d/`에 추가했다. 원본 SQL의 마지막 CREATE TABLE을 `INSERT INTO mlops.ineligible_loan_model_features_target`으로 변경하고 `base_ym` 컬럼(`DATE_FORMAT(applicant_date, '%Y%m')`)을 추가했다.

**핵심 교훈**
MariaDB `docker-entrypoint-initdb.d`는 최초 볼륨 초기화 시에만 실행된다. 스크립트 누락 시 볼륨을 삭제하고 재시작해야 반영된다(`docker-compose down -v`).

---

## 6. API 컨테이너 기동 실패 — `dill` 모듈 없음

**에러 메시지**
```
ModuleNotFoundError: No module named 'dill'
ERROR: Application startup failed. Exiting.
```

**원인**
MLflow 모델 로딩 시 내부적으로 `dill`을 사용하는데, `api/requirements.txt`에 `dill`이 없어서 API 컨테이너 기동 시 import 오류 발생.

**해결 방법**
`api/requirements.txt`에 `dill`을 추가하고 이미지를 재빌드했다.

```
dill
```

**핵심 교훈**
MLflow가 모델을 직렬화/역직렬화할 때 `dill`, `cloudpickle` 등 추가 의존성을 요구할 수 있다. 학습 환경과 서빙 환경의 패키지를 일치시킬 것.

---

## 7. API 내부 절대 임포트 오류 — `from api.*`

**에러 메시지**
```
ModuleNotFoundError: No module named 'api'
```

**원인**
`api/main.py`, `api/routers/predict.py`, `api/services/preprocess_service.py`에서 `from api.routers`, `from api.schemas` 등 절대 경로 임포트를 사용했다. Docker 컨테이너 내부에서는 `/app`이 루트이므로 `api`라는 패키지가 존재하지 않아 임포트 실패.

**해결 방법**
`from api.*` → `from routers.*`, `from schemas.*`, `from services.*` 형태의 상대 임포트로 변경했다.

```python
# Before
from api.routers import health, predict
from api.schemas.request import LoanApplicantRequest

# After
from routers import health, predict
from schemas.request import LoanApplicantRequest
```

**핵심 교훈**
Docker 컨테이너에서 FastAPI 앱을 실행할 때 WORKDIR를 앱 루트(`/app`)로 설정하면, 패키지 내부 임포트는 절대 경로가 아닌 상대 경로(또는 WORKDIR 기준)로 작성해야 한다.

---

## 8. Nginx 502 Bad Gateway — 컨테이너 재생성 후 upstream IP 캐싱

**에러 메시지**
```
502 Bad Gateway
connect() failed (111: Connection refused) while connecting to upstream: http://172.22.0.5:8000/predict
```

**원인**
Nginx는 시작 시점에 upstream 호스트명(`api`)을 한 번만 DNS 조회해 IP를 캐싱한다. API 컨테이너가 재생성되면 새 IP를 할당받지만 Nginx는 기존 IP로 계속 요청해 Connection refused가 발생한다.

**해결 방법**
`nginx.conf`에 Docker 내부 DNS resolver(`127.0.0.11`)를 추가하고, upstream을 변수로 선언해 요청마다 DNS를 재조회하도록 설정한다.

```nginx
location / {
    resolver 127.0.0.11 valid=10s;
    set $upstream http://api:8000;
    proxy_pass $upstream;
    ...
}
```

**핵심 교훈**
Docker Compose 환경에서 Nginx upstream을 고정 블록(`upstream {}`)으로 선언하면 IP 캐싱 문제가 발생한다. `resolver 127.0.0.11`과 변수(`set $upstream`)를 사용하면 컨테이너 재생성 시에도 DNS를 동적으로 재조회한다.
