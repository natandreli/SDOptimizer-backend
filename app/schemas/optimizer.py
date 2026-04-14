from typing import Dict, List, Tuple

from pydantic import BaseModel


class OptimizationConfigSchema(BaseModel):
    parameter_names: List[str]
    initial_values: List[float]
    bounds: List[Tuple[float, float]]

    rho_factors: List[float]
    epsilon: float
    max_runs: int

    target_variable: str
    statistic: str


class OptimizationHistorySchema(BaseModel):
    rewards: List[float]
    best_rewards: List[float]
    parameters: List[List[float]]
    actions: List[Tuple[int, ...]]


class OptimizationResultSchema(BaseModel):
    best_parameters: Dict[str, float]
    best_score: float
    history: OptimizationHistorySchema


class OptimizationResponse(BaseModel):
    result: OptimizationResultSchema
