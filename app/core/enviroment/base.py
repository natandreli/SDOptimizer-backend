from abc import ABC, abstractmethod
from typing import List


class BaseEnvironment(ABC):
    """
    Abstract Base Class for Reinforcement Learning environments.
    """

    @abstractmethod
    def step(self, parameters: List[float]) -> float:
        """
        Apply parameters to the environment and return the resulting reward.

        In the context of System Dynamics optimization, one step usually
        corresponds to a full simulation run.

        Args:
            parameters: List of parameter values to evaluate.

        Returns:
            float: Scalar reward value.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the environment to its initial state.
        """
        pass
