from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.models import (
    GetModelResponse,
    UploadModelResponse,
)
from app.schemas.optimizer import OptimizationOptionsSchema, OptimizationResultSchema
from app.schemas.simulation import SimulationOptionsSchema, SimulationResultSchema

__all__ = [
    "GetModelResponse",
    "OptimizationOptionsResponse",
    "OptimizationResponse",
    "SimulationOptionsResponse",
    "SimulationResponse",
    "UploadModelResponse",
]


class SimulationResponse(BaseModel):
    result: SimulationResultSchema = None


class OptimizationResponse(BaseModel):
    result: Optional[OptimizationResultSchema] = None
    results: List[OptimizationResultSchema] = Field(default_factory=list)
    best_optimization_number: Optional[int] = None
    total_execution_time_ms: float = 0.0


class OptimizationOptionsResponse(BaseModel):
    options: OptimizationOptionsSchema


class SimulationOptionsResponse(BaseModel):
    options: SimulationOptionsSchema
