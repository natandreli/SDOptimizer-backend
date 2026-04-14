from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..agent.e_greedy_agent import EGreedyBanditAgent

ACTION_DIRECTIONS = [1, -1, 0]


class ModelOptimizer:
    """
    Parameter optimizer for System Dynamics models using an ε-greedy
    multi-armed bandit (MAB) strategy.
    """

    def __init__(
        self,
        wrapper,
        parameter_names: List[str],
        initial_values: List[float],
        bounds: List[Tuple[float, float]],
        rho_factors: List[float],
        agent: EGreedyBanditAgent,
        objective_fn: Callable,
        max_runs: int = 500,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            wrapper: Instance of PySDWrapper used to execute simulations.
            parameter_names: Names of the parameters to optimize.
            initial_values: Initial values for each parameter.
            bounds: List of (min, max) constraints per parameter.
            rho_factors: Relative step sizes per parameter.
            agent: ε-greedy bandit agent.
            objective_fn: Function mapping simulation results → scalar reward.
            max_runs: Number of optimization iterations.

        Raises:
            ValueError: If input dimensions are inconsistent.
        """
        if len(parameter_names) != len(initial_values):
            raise ValueError(
                "parameter_names and initial_values must have the same length."
            )
        if len(bounds) != len(parameter_names):
            raise ValueError("bounds must match number of parameters.")
        if len(rho_factors) != len(parameter_names):
            raise ValueError("rho_factors must match number of parameters.")

        self.wrapper = wrapper
        self.parameter_names = parameter_names
        self.current_parameters = list(initial_values)
        self.bounds = bounds
        self.rho_factors = rho_factors
        self.agent = agent
        self.objective_fn = objective_fn
        self.max_runs = max_runs

        self.history: Dict[str, List[Any]] = {
            "rewards": [],
            "best_rewards": [],
            "parameters": [],
            "actions": [],
        }

    def optimize(self) -> Tuple[List[float], float]:
        """
        Execute the ε-greedy optimization loop.

        Returns:
            Tuple containing:
                - best_params: Best parameter configuration found
                - best_score: Corresponding objective value
        """
        best_score = -np.inf
        best_params = list(self.current_parameters)

        for _ in range(self.max_runs):
            prev_params = list(self.current_parameters)
            action = self.agent.select_action()
            self._apply_action(action)

            if self._is_feasible(self.current_parameters):
                reward = self._evaluate(self.current_parameters)
                self.agent.update(action, reward)

                if reward > best_score:
                    best_score = reward
                    best_params = list(self.current_parameters)

            else:
                reward = -100.0
                self.agent.update(action, reward)
                self.current_parameters = prev_params

            self.history["rewards"].append(reward)
            self.history["best_rewards"].append(best_score)
            self.history["parameters"].append(list(self.current_parameters))
            self.history["actions"].append(action)

        return best_params, best_score

    def get_history(self) -> Dict[str, List[Any]]:
        """
        Retrieve optimization trajectory data.

        Returns:
            Dictionary containing:
                - rewards: Reward at each iteration
                - best_rewards: Best reward found up to each iteration
                - parameters: Parameter values at each step
                - actions: Actions taken at each iteration
        """
        return self.history

    def _apply_action(self, action: tuple) -> None:
        """
        Apply a relative update to parameters based on selected action.

        Args:
            action: Tuple representing action indices for each parameter.
        """
        for i, action_idx in enumerate(action):
            direction = ACTION_DIRECTIONS[action_idx]
            if direction != 0:
                self.current_parameters[i] *= 1 + direction * self.rho_factors[i]

    def _is_feasible(self, params: List[float]) -> bool:
        """
        Check whether parameter values satisfy feasibility constraints.

        Args:
            params: List of parameter values.

        Returns:
            True if all parameters are finite and within bounds, False otherwise.
        """
        for value, (low, high) in zip(params, self.bounds):
            if not np.isfinite(value):
                return False
            if low is not None and value < low:
                return False
            if high is not None and value > high:
                return False
        return True

    def _evaluate(self, params: List[float]) -> float:
        """
        Execute simulation and compute objective value.

        Args:
            params: Parameter values to evaluate.

        Returns:
            Scalar reward value (objective function output).
        """
        overrides = dict(zip(self.parameter_names, params))
        return self.wrapper.run_and_evaluate(overrides, self.objective_fn)
