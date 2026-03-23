from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimulationConfigSchema(BaseModel):
    """Configuration for running a simulation."""

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
    """Result of a simulation run."""

    time_series: Dict[str, List[float]] = Field(
        description="Time-series data for each variable. Keys are variable names.",
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
