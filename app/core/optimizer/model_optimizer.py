import numpy as np
from typing import Any, Dict, List, Tuple, Callable
from ..agent.e_greedy_agent import EGreedyAgent
from ..enviroment.base import BaseEnvironment


class ModelOptimizer:
    """
    Parameter optimizer for System Dynamics models using an ε-greedy
    multi-armed bandit (MAB) strategy.
    """

    ACTION_MAP = {0: 1, 1: -1, 2: 0}  # 0: Increase, 1: Decrease, 2: Maintain

    def __init__(
        self,
        environment: BaseEnvironment,
        agent: EGreedyAgent,
        parameter_names: List[str],
        initial_values: List[float],
        bounds: List[Tuple[float, float]],
        rho_factors: List[float],
        max_runs: int = 100,
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            environment: SD environment to evaluate configurations.
            agent: ε-greedy bandit agent.
            parameter_names: Names of parameters to optimize.
            initial_values: Initial parameter values.
            bounds: List of (min, max) tuples for each parameter.
            rho_factors: Relative step sizes (ρ) for each parameter.
            max_runs: Number of iterations to execute.
        """
        self.env = environment
        self.agent = agent
        self.parameter_names = parameter_names
        self.current_params = list(initial_values)
        self.bounds = bounds
        self.rho_factors = rho_factors
        self.max_runs = max_runs

        self.history: Dict[str, List[Any]] = {
            "rewards": [],
            "best_rewards": [],
            "parameters": [],
            "actions": [],
        }


    def optimize(self) -> Tuple[List[float], float]:
        current_reward = self.env.step(self.current_params)
        best_params = list(self.current_params)
        best_score = current_reward

        for i in range(self.max_runs):
            action = self.agent.select_action()
            directions = [self.ACTION_MAP[idx] for idx in action]
            
            trial_params = [
                val * (1 + d * rho)
                for val, d, rho in zip(self.current_params, directions, self.rho_factors)
            ]

            if self._is_feasible(trial_params):
                new_reward = self.env.step(trial_params)
                
                reward = new_reward
                self.agent.update(action, reward)

                if new_reward > current_reward:
                    self.current_params = list(trial_params)
                    current_reward = new_reward
                    
                    if current_reward > best_score:
                        best_score = current_reward
                        best_params = list(self.current_params)
            else:
                reward = -100.0
                self.agent.update(action, reward)

            self.history["rewards"].append(reward)
            self.history["best_rewards"].append(best_score)
            self.history["parameters"].append(list(self.current_params))
            self.history["actions"].append(action)

        return best_params, best_score

    def get_history(self) -> Dict[str, List[Any]]:
        """Return optimization execution history."""
        return self.history

    def _is_feasible(self, params: List[float]) -> bool:
        """Check if parameters are within bounds and are finite."""
        for val, (low, high) in zip(params, self.bounds):
            if not np.isfinite(val):
                return False
            if low is not None and val < low:
                return False
            if high is not None and val > high:
                return False
        return True