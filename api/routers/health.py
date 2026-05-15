from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/model/info")
async def model_info(request: Request):
    svc = request.app.state.model_service
    return {
        "model_name": svc.model_name,
        "model_stage": svc.model_stage,
        "model_version": svc.model_version,
    }
