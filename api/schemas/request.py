from typing import Literal
from pydantic import BaseModel, Field


class LoanApplicantRequest(BaseModel):
    applicant_id: str = Field(..., examples=["LP001002"])
    gender: Literal["Male", "Female"]
    married: Literal["Yes", "No"]
    family_dependents: Literal["0", "1", "2", "3+"]
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    applicant_income: int = Field(..., gt=0)
    coapplicant_income: int = Field(..., ge=0)
    loan_amount_term: int = Field(..., gt=0, examples=[60])
    credit_history: float = Field(..., ge=0.0, le=1.0)
    property_area: Literal["Rural", "Semiurban", "Urban"]
