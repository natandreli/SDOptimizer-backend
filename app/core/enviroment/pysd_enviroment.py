from typing import List

from .base import BaseEnvironment


class PySDEnvironment(BaseEnvironment):
    """
    System Dynamics environment using PySD for simulation.
    """

    def __init__(self, parser, objective_fn, param_names: List[str]):
        """
        Initialize the PySD environment.

        Args:
            parser: PySDParser instance to run simulations.
            objective_fn: Callable mapping simulation results to a scalar reward.
            param_names: List of parameter names in the same order as they
                appear in the parameters list passed to 'step'.
        """
        self.parser = parser
        self.objective_fn = objective_fn
        self.param_names = param_names
        

    def step(self, parameters: List[float]) -> float:
        overrides = dict(zip(self.param_names, parameters))
        results = self.parser.run(overrides)
        return self.objective_fn(results)


    def reset(self) -> None:
        """
        Reset the environment. For MAB, this is typically a no-op.
        """
        pass