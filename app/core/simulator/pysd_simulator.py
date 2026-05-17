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
        # Determine output recording interval (target around 1000 points max for smooth chart rendering and zero JSON overhead)
        total_time = self.config.total_time
        dt = self.config.dt
        
        if total_time / dt > 1000:
            steps_per_output = max(1, int((total_time / 1000) / dt))
            output_step = steps_per_output * dt
        else:
            output_step = dt

        return_timestamps = np.arange(0, total_time + output_step, output_step)
        # Identify all stateful variables (stocks) in the model doc to handle stock initial overrides dynamically
        stateful_vars: dict[str, str] = {}
        try:
            doc = self.model.doc
            if doc is not None and not doc.empty:
                for _, row in doc.iterrows():
                    element_type = str(row.get("Type", "")).strip().lower()
                    if element_type == "stateful":
                        real_name = str(row.get("Real Name", "")).strip()
                        py_name = str(row.get("Py Name", "")).strip()
                        stateful_vars[real_name] = py_name
                        stateful_vars[py_name] = py_name
        except Exception:
            pass

        params = {}
        stock_overrides = {}
        if self.config.parameter_overrides:
            for k, v in self.config.parameter_overrides.items():
                pysd_name = k.replace(" ", "_")
                if k in stateful_vars or pysd_name in stateful_vars:
                    # Map to the proper Py Name for PySD to accept it
                    key = stateful_vars.get(k, stateful_vars.get(pysd_name, pysd_name))
                    stock_overrides[key] = v
                else:
                    params[k] = v

        # Override integration step size in PySD's components.Time manager to enforce the user's requested step size.
        original_time_step_func = getattr(self.model.time, "time_step", None)
        if self.config.dt is not None:
            self.model.time.time_step = lambda: self.config.dt

        run_kwargs = {}
        if stock_overrides:
            # Pass custom stock overrides via initial_condition!
            run_kwargs["initial_condition"] = (0, stock_overrides)

        try:
            df = self.model.run(
                params=params if params else None,
                return_columns=self.config.return_columns,
                return_timestamps=return_timestamps,
                **run_kwargs
            )
        finally:
            if original_time_step_func is not None:
                self.model.time.time_step = original_time_step_func
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
        """
        Extract parameter names from the model documentation.

        Returns:
            Set of parameter names (both real and PySD names) found in the model documentation.
        """
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
        """
        Compute mean, min, max, initial, final for each variable's time series.

        Args:
            time_series: A dictionary mapping variable names to their time series values.

        Returns:
            A dictionary mapping variable names to their summary statistics.
        """
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
