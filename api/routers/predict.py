from fastapi import APIRouter, Request
from api.schemas.request import LoanApplicantRequest
from api.schemas.response import PredictResponse

router = APIRouter()


@router.post("", response_model=PredictResponse)
async def predict(request: Request, body: LoanApplicantRequest):
    preprocess_svc = request.app.state.preprocess_service
    model_svc = request.app.state.model_service

    features = preprocess_svc.transform(body)
    result = model_svc.predict(features)

    return PredictResponse(
        applicant_id=body.applicant_id,
        predict=result["predict"],
        probability=result["probability"],
        model_version=model_svc.model_version,
    )


@router.post("/batch", response_model=list[PredictResponse])
async def predict_batch(request: Request, bodies: list[LoanApplicantRequest]):
    preprocess_svc = request.app.state.preprocess_service
    model_svc = request.app.state.model_service

    results = []
    for body in bodies:
        features = preprocess_svc.transform(body)
        result = model_svc.predict(features)
        results.append(PredictResponse(
            applicant_id=body.applicant_id,
            predict=result["predict"],
            probability=result["probability"],
            model_version=model_svc.model_version,
        ))
    return results
