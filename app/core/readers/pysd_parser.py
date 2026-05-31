from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pysd


class PySDParser:
    """
    Parser for executing System Dynamics models using PySD and evaluating objective functions.

    This class abstracts:
    - Model loading (.mdl or .py)
    - Parameter management and overrides
    - Simulation execution
    - Objective evaluation from simulation outputs

    It is designed to be used in pipelines where models are dynamically loaded
    (e.g., after file upload) and evaluated multiple times with different parameter sets.
    """

    def __init__(
        self,
        model_path_or_obj: str | pysd.PySD,
        parameters: List[Dict[str, Any]],
    ) -> None:
        """
        Initialize the wrapper by loading the model and configuring parameter mappings.

        Args:
            model_path_or_obj: Path to the model file (.mdl or .py) or an already loaded pysd.PySD object.
            parameters: List of parameter metadata dictionaries. Each dictionary must contain:
                - "name" (str): Original parameter name
                - "initial_value" (float): Default value for simulation
                - "min" (Optional[float]): Minimum allowed value
                - "max" (Optional[float]): Maximum allowed value

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If required parameter fields are missing.
        """
        if hasattr(model_path_or_obj, "run"):
            self.model = model_path_or_obj
        elif str(model_path_or_obj).lower().endswith(".mdl"):
            self.model = pysd.read_vensim(str(model_path_or_obj))
        else:
            self.model = pysd.load(str(model_path_or_obj))

        self.original_parameters = parameters

        self.params_map: Dict[str, str] = {
            p["name"]: p["name"].replace(" ", "_") for p in parameters
        }

        self.initial_values: Dict[str, float] = {
            self.params_map[p["name"]]: p["initial_value"] for p in parameters
        }

        self.param_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {
            self.params_map[p["name"]]: (
                p.get("min"),
                p.get("max"),
            )
            for p in parameters
        }

        # Identify all stateful variables (stocks) in the model doc to handle stock initial overrides dynamically
        self.stateful_vars: Dict[str, str] = {}
        try:
            doc = self.model.doc
            if doc is not None and not doc.empty:
                for _, row in doc.iterrows():
                    element_type = str(row.get("Type", "")).strip().lower()
                    if element_type == "stateful":
                        real_name = str(row.get("Real Name", "")).strip()
                        py_name = str(row.get("Py Name", "")).strip()
                        self.stateful_vars[real_name] = py_name
                        self.stateful_vars[py_name] = py_name
        except Exception:
            pass

    def run(
        self,
        overrides: Optional[Dict[str, float]] = None,
        return_columns: Optional[List[str]] = None,
        dt: Optional[float] = None,
        total_time: Optional[float] = None,
        **run_kwargs,
    ) -> pd.DataFrame:
        """
        Execute the simulation with optional parameter overrides.

        Args:
            overrides: Dictionary of parameter values to override. Keys may be either:
                - Original parameter names (with spaces)
                - PySD-compatible names (underscored)

        Returns:
            pd.DataFrame: Simulation results where:
                - Index represents simulation time
                - Columns represent model variables

        Raises:
            ValueError: If any parameter is invalid or out of bounds.
        """
        params = self.initial_values.copy()
        stock_overrides = {}

        if overrides:
            self.validate_overrides(overrides)
            for name, value in overrides.items():
                if not np.isfinite(value):
                    raise ValueError(f"'{name}' has invalid value: {value}")

                pysd_name = self.params_map.get(name, name.replace(" ", "_"))

                # Check if this name refers to a stateful variable (stock)
                if name in self.stateful_vars or pysd_name in self.stateful_vars:
                    # Map to the proper Py Name for PySD to accept it
                    key = self.stateful_vars.get(
                        name, self.stateful_vars.get(pysd_name, pysd_name)
                    )
                    stock_overrides[key] = value
                else:
                    params[pysd_name] = value

        full_kwargs = {
            "params": params,
        }
        # Merge extra run kwargs (e.g. return_timestamps from Turbo Mode)
        full_kwargs.update(run_kwargs)

        if stock_overrides:
            # Pass custom stock overrides via initial_condition!
            full_kwargs["initial_condition"] = (0, stock_overrides)

        if dt is not None and total_time is not None:
            # Only generate default timestamps if not already provided
            if "return_timestamps" not in full_kwargs:
                full_kwargs["return_timestamps"] = np.arange(0, total_time + dt, dt)

        if return_columns is not None:
            full_kwargs["return_columns"] = return_columns

        # Override integration step size in PySD's components.Time manager to enforce user's input dt.
        # This drastically reduces the number of steps and solves the PySD simulation speed bottleneck!
        original_time_step_func = getattr(self.model.time, "time_step", None)
        if dt is not None:
            self.model.time.time_step = lambda: dt

        try:
            return self.model.run(**full_kwargs)
        finally:
            if original_time_step_func is not None:
                self.model.time.time_step = original_time_step_func

    def validate_overrides(self, overrides: Dict[str, float]) -> None:
        """
        Validate that override parameters exist and are within defined bounds.

        Args:
            overrides: Dictionary of parameter values to validate.

        Raises:
            ValueError: If:
                - A parameter does not exist in the model
                - A value is outside its allowed range (min/max)
        """
        for name, value in overrides.items():
            pysd_name = self.params_map.get(name, name.replace(" ", "_"))

            # If it's a stateful variable (stock), it is valid!
            if name in self.stateful_vars or pysd_name in self.stateful_vars:
                continue

            if pysd_name not in self.param_bounds:
                raise ValueError(f"Unknown parameter: '{name}'")

            min_v, max_v = self.param_bounds[pysd_name]

            if min_v is not None and value < min_v:
                raise ValueError(f"'{name}'={value} is below minimum ({min_v})")

            if max_v is not None and value > max_v:
                raise ValueError(f"'{name}'={value} is above maximum ({max_v})")
