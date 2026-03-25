from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class SimulationConfigSchema(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dt": 0.25,
                    "total_time": 100.0,
                    "parameter_overrides": {"cash revenue": 1500.0},
                }
            ]
        }
    )

    dt: float = Field(
        default=0.25,
        gt=0,
        description="Time step for Euler integration.",
    )
    total_time: float = Field(
        default=100.0,
        gt=0,
        description="Total simulation time.",
    )
    parameter_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Override parameter values. Keys are parameter real names "
            "(e.g. 'cash revenue'), values are the new numeric values."
        ),
    )


class SimulationResultSchema(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "time_series": {"Variable 1": [10.0, 11.0, 12.0]},
                    "parameter_series": {"Param 1": [0.5, 0.5, 0.5]},
                    "summary_stats": {
                        "Variable 1": {
                            "mean": 11.0,
                            "min": 10.0,
                            "max": 12.0,
                            "initial": 10.0,
                            "final": 12.0,
                        }
                    },
                    "steps_executed": 401,
                    "config": {
                        "dt": 0.25,
                        "total_time": 100.0,
                        "parameter_overrides": {},
                    },
                }
            ]
        }
    )

    time_series: Dict[str, List[float]] = Field(
        description=(
            "Time-series data for model variables (excluding parameters). "
            "Keys are variable names."
        ),
    )
    parameter_series: Dict[str, List[float]] = Field(
        default_factory=dict,
        description=(
            "Time-series data for model parameters/constants. Keys are parameter names."
        ),
    )
    summary_stats: Dict[str, Dict[str, float]] = Field(
        description=(
            "Summary statistics per variable: mean, min, max, initial, final."
        ),
    )
    steps_executed: int = Field(description="Number of simulation steps executed.")
    config: SimulationConfigSchema = Field(
        description="Configuration used for this simulation run.",
    )
