from typing import Annotated, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, StrictInt


class OptimizationConfigSchema(BaseModel):
    parameter_names: List[str]
    initial_values: List[float]
    bounds: List[Tuple[float, float]]

    rho_factors: List[float]
    epsilon: float
    max_runs: int
    optimization_count: Annotated[StrictInt, Field(ge=1, le=100)] = 1

    target_variable: str
    statistic: Literal["final", "mean", "max", "min"]
    direction: Literal["maximize", "minimize"] = "maximize"

    dt: Optional[float] = None
    total_time: Optional[float] = None
    final_time: Optional[float] = None


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
    dt: Optional[float] = None
    total_time: Optional[float] = None
    time_unit: str = ""


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


class ParameterChangeSchema(BaseModel):
    initial_value: float
    optimized_value: float
    change_percentage: float


class OptimizationConfigSummarySchema(BaseModel):
    target_variable: str
    statistic: str
    direction: str
    max_runs: int
    epsilon: float


class OptimizationResultSchema(BaseModel):
    optimization_number: int = 1
    execution_time_ms: float = 0.0
    best_parameters: Dict[str, float]
    best_score: float
    history: OptimizationHistorySchema
    initial_parameters: Dict[str, float]
    initial_score: float
    improvement_percentage: float
    parameter_changes: Dict[str, ParameterChangeSchema]
    config_summary: OptimizationConfigSummarySchema
    steps_per_simulation: int = 0
    total_mathematical_steps: int = 0
