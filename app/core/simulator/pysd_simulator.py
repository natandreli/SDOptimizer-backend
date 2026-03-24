from __future__ import annotations

import numpy as np
import pysd

from app.schemas.simulation import SimulationConfigSchema, SimulationResultSchema


class PySDSimulator:
    """
    Simulator that uses PySD natively to run System Dynamics models.
    """

    def __init__(self, model: pysd.PySD, config: SimulationConfigSchema):
        self.model = model
        self.config = config

    def simulate(self) -> SimulationResultSchema:
        """
        Run the simulation using natively PySD model.run().

        Returns:
            SimulationResultSchema container with results.
        """
        return_timestamps = np.arange(
            0, self.config.total_time + self.config.dt, self.config.dt
        )
        params = (
            self.config.parameter_overrides if self.config.parameter_overrides else None
        )
        df = self.model.run(
            params=params,
            return_columns=None,
            return_timestamps=return_timestamps,
        )
        time_series = {col: df[col].tolist() for col in df.columns}
        time_list = df.index.tolist()
        if "Time" not in time_series:
            time_series["Time"] = time_list
        if "time" not in time_series:
            time_series["time"] = time_list

        summary_stats = self._compute_summary_stats(time_series)

        return SimulationResultSchema(
            time_series=time_series,
            summary_stats=summary_stats,
            steps_executed=len(return_timestamps),
            config=self.config,
        )

    @staticmethod
    def _compute_summary_stats(
        time_series: dict[str, list[float]],
    ) -> dict[str, dict[str, float]]:
        """Compute mean, min, max, initial, final for each variable's time series."""
        stats: dict[str, dict[str, float]] = {}
        for name, values in time_series.items():
            if not values:
                continue
            arr = np.array(values, dtype=float)
            stats[name] = {
                "mean": float(np.nanmean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "final": float(arr[-1]),
                "initial": float(arr[0]),
            }
        return stats
