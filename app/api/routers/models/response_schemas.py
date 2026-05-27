from typing import Optional

from pydantic import BaseModel

from app.schemas.models import (
    GetModelResponse,
    ModelSchema,
    UploadModelResponse,
)
from app.schemas.optimizer import OptimizationOptionsSchema, OptimizationResultSchema
from app.schemas.simulation import SimulationOptionsSchema, SimulationResultSchema

class SimulationResponse(BaseModel):
    result: SimulationResultSchema = None


class OptimizationResponse(BaseModel):
    result: OptimizationResultSchema = None


class OptimizationOptionsResponse(BaseModel):
    options: OptimizationOptionsSchema


class SimulationOptionsResponse(BaseModel):
    options: SimulationOptionsSchema
