from typing import Optional

from pydantic import BaseModel

from app.schemas import (
    ModelSchema,
    SimulationResultSchema,
)
from app.schemas.optimizer import OptimizationOptionsSchema, OptimizationResultSchema
from app.schemas.simulation import SimulationOptionsSchema


class UploadModelResponse(BaseModel):
    model_id: str
    model: Optional[ModelSchema] = None


class GetModelResponse(BaseModel):
    model_id: str
    model: Optional[ModelSchema] = None


class SimulationResponse(BaseModel):
    result: SimulationResultSchema = None


class OptimizationResponse(BaseModel):
    result: OptimizationResultSchema = None


class OptimizationOptionsResponse(BaseModel):
    options: OptimizationOptionsSchema


class SimulationOptionsResponse(BaseModel):
    options: SimulationOptionsSchema
