from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import health, predict
from services.model_service import ModelService
from services.preprocess_service import PreprocessService

model_service = ModelService()
preprocess_service = PreprocessService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    preprocess_service.load()
    yield


app = FastAPI(
    title="Ineligible Loan Model API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])

app.state.model_service = model_service
app.state.preprocess_service = preprocess_service
