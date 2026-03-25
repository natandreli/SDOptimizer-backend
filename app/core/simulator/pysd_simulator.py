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
        time_series.pop("Time", None)
        time_series["time"] = time_list

        parameter_names = self._get_parameter_names()
        parameter_series = {
            name: values
            for name, values in time_series.items()
            if name in parameter_names and not self._is_control_variable(name)
        }
        variable_series = {
            name: values
            for name, values in time_series.items()
            if name not in parameter_names and not self._is_control_variable(name)
        }
        summary_series = {
            name: values for name, values in time_series.items() if name != "time"
        }
        summary_stats = self._compute_summary_stats(summary_series)

        return SimulationResultSchema(
            time_series=variable_series,
            parameter_series=parameter_series,
            summary_stats=summary_stats,
            steps_executed=len(return_timestamps),
            config=self.config,
        )

    def _get_parameter_names(self) -> set[str]:
        """Return parameter (constant) names from model documentation."""
        parameter_names: set[str] = set()
        try:
            doc = self.model.doc
            if doc is None or doc.empty:
                return parameter_names

            for _, row in doc.iterrows():
                element_type = str(row.get("Type", "")).strip().lower()
                if element_type == "constant":
                    real_name = str(row.get("Real Name", "")).strip()
                    py_name = str(row.get("Py Name", "")).strip()
                    if real_name:
                        parameter_names.add(real_name)
                    if py_name:
                        parameter_names.add(py_name)
        except Exception:
            return set()

        return parameter_names

    @staticmethod
    def _is_control_variable(name: str) -> bool:
        normalized = " ".join(name.strip().lower().replace("_", " ").split())
        return normalized in {
            "time",
            "saveper",
            "time step",
            "initial time",
            "final time",
        }

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
