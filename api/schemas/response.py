from pydantic import BaseModel


class PredictResponse(BaseModel):
    applicant_id: str
    predict: int
    probability: float
    model_version: str | None = None
