from typing import Optional

from pydantic import BaseModel

from app.schemas import (
    ModelSchema,
    SimulationResultSchema,
)


class UploadModelResponse(BaseModel):
    model_id: str
    model: Optional[ModelSchema] = None


class GetModelResponse(BaseModel):
    model_id: str
    model: Optional[ModelSchema] = None


class SimulationResponse(BaseModel):
    result: SimulationResultSchema
