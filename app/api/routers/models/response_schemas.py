from typing import Optional

from pydantic import BaseModel

from app.schemas import (
    ModelInformationSchema,
    SimulationResultSchema,
    ValidationResultSchema,
)


class UploadModelResponse(BaseModel):
    model_id: str
    validation: ValidationResultSchema
    model_info: Optional[ModelInformationSchema] = None


class SimulationResponse(BaseModel):
    result: SimulationResultSchema
