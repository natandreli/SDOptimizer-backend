from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel


class OptimizationConfigSchema(BaseModel):
    parameter_names: List[str]
    initial_values: List[float]
    bounds: List[Tuple[float, float]]

    rho_factors: List[float]
    epsilon: float
    max_runs: int

    target_variable: str
    statistic: Literal["final", "mean", "max", "min"]
    direction: Literal["maximize", "minimize"] = "maximize"


class OptimizationParameterOptionSchema(BaseModel):
    name: str
    initial_value: float
    suggested_bounds: Tuple[float, float]
    suggested_rho_factor: float = 0.01


class OptimizationDefaultsSchema(BaseModel):
    epsilon: float = 0.7
    max_runs: int = 200
    statistic: Literal["final", "mean", "max", "min"] = "max"
    direction: Literal["maximize", "minimize"] = "maximize"


class OptimizationOptionsSchema(BaseModel):
    parameters: List[OptimizationParameterOptionSchema]
    target_variables: List[str]
    statistics: List[Literal["final", "mean", "max", "min"]]
    directions: List[Literal["maximize", "minimize"]]
    defaults: OptimizationDefaultsSchema


class OptimizationHistorySchema(BaseModel):
    rewards: List[float]
    best_rewards: List[float]
    parameters: List[List[float]]
    actions: List[Tuple[int, ...]]


class OptimizationResultSchema(BaseModel):
    best_parameters: Dict[str, float]
    best_score: float
    history: OptimizationHistorySchema
