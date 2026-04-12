import pysd
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple


class PySDWrapper:
    """
    Wrapper for executing System Dynamics models using PySD and evaluating objective functions.

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
        model_path: str,
        parameters: List[Dict[str, Any]],
    ) -> None:
        """
        Initialize the wrapper by loading the model and configuring parameter mappings.

        Args:
            model_path: Path to the model file. Supported formats:
                - .mdl (Vensim model)
                - .py (PySD-translated model)

            parameters: List of parameter metadata dictionaries. Each dictionary must contain:
                - "name" (str): Original parameter name
                - "initial_value" (float): Default value for simulation
                - "min" (Optional[float]): Minimum allowed value
                - "max" (Optional[float]): Maximum allowed value

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If required parameter fields are missing.
        """
        if model_path.lower().endswith(".mdl"):
            self.model = pysd.read_vensim(model_path)
        else:
            self.model = pysd.load(model_path)

        self.original_parameters = parameters

        self.params_map: Dict[str, str] = {
            p["name"]: p["name"].replace(" ", "_")
            for p in parameters
        }

        self.initial_values: Dict[str, float] = {
            self.params_map[p["name"]]: p["initial_value"]
            for p in parameters
        }

        self.param_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {
            self.params_map[p["name"]]: (
                p.get("min"),
                p.get("max"),
            )
            for p in parameters
        }

    def run(self, overrides: Optional[Dict[str, float]] = None) -> pd.DataFrame:
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

        if overrides:
            self.validate_overrides(overrides)
            if not np.isfinite(value):
                raise ValueError(f"'{name}' has invalid value: {value}")

            for name, value in overrides.items():
                pysd_name = self.params_map.get(name, name.replace(" ", "_"))
                params[pysd_name] = value

        return self.model.run(params=params)

    def evaluate(
        self,
        results: pd.DataFrame,
        objective_fn: Callable[[pd.DataFrame], float],
    ) -> float:
        """
        Evaluate a scalar objective function on simulation results.

        Args:
            results: DataFrame returned by `run`, containing time-series data
                for all model variables.

            objective_fn: Function that takes the results DataFrame and returns
                a scalar numeric value.
                
        Returns:
            float: Scalar value computed by the objective function.

        Raises:
            ValueError: If the objective function does not return a numeric value.
            KeyError: If the objective function accesses a non-existent column.
        """
        value = objective_fn(results)

        if not isinstance(value, (int, float, np.floating)):
            raise ValueError("Objective function must return a numeric scalar value.")

        return float(value)

    def run_and_evaluate(
        self,
        overrides: Optional[Dict[str, float]],
        objective_fn: Callable[[pd.DataFrame], float],
    ) -> float:
        """
        Execute the simulation and evaluate the objective function in a single step.

        Args:
            overrides: Dictionary of parameter overrides to apply during simulation.
            objective_fn: Function that evaluates the simulation output.

        Returns:
            float: Scalar objective value.

        Raises:
            ValueError: If parameters or objective function are invalid.
        """
        results = self.run(overrides)
        return self.evaluate(results, objective_fn)

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

            if pysd_name not in self.param_bounds:
                raise ValueError(f"Unknown parameter: '{name}'")

            min_v, max_v = self.param_bounds[pysd_name]

            if min_v is not None and value < min_v:
                raise ValueError(f"'{name}'={value} is below minimum ({min_v})")

            if max_v is not None and value > max_v:
                raise ValueError(f"'{name}'={value} is above maximum ({max_v})")