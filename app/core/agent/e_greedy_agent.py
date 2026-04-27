import random
from typing import Tuple

import numpy as np


class EGreedyAgent:
    """
    ε-greedy multi-armed bandit agent.

    Maintains action-value estimates (Q-values) and updates them using
    incremental averaging based on observed rewards.
    """

    def __init__(
        self,
        action_shape: Tuple[int, ...],
        epsilon: float,
        epsilon_min: float = 0.01,
        optimistic_init: float = 0.0,
    ) -> None:
        """
        Initialize the agent.

        Args:
            action_shape: Shape of the action space.
                Example for v parameters: (3, 3, ..., 3)
            epsilon: Initial exploration probability in [0, 1].
            epsilon_min: Minimum exploration probability.
            optimistic_init: Initial value for Q-values to encourage exploration.

        Raises:
            ValueError: If epsilon values are outside valid range.
        """
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not (0.0 <= epsilon_min <= 1.0):
            raise ValueError(
                f"epsilon_min must be in [0, 1], got {epsilon_min}"
            )

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.q_table = np.full(action_shape, float(optimistic_init), dtype=float)
        self.n_table = np.zeros(action_shape, dtype=int)

    def select_action(self) -> Tuple[int, ...]:
        """
        Select an action using ε-greedy policy.

        Returns:
            Tuple representing selected action indices.
        """
        if random.random() < self.epsilon:
            return self._random_action()
        return self._greedy_action()

    def _greedy_action(self) -> Tuple[int, ...]:
        max_val = np.max(self.q_table)
        candidates = np.argwhere(self.q_table == max_val)
        chosen = candidates[random.randrange(len(candidates))]
        return tuple(chosen.tolist())

    def _random_action(self) -> Tuple[int, ...]:
        """Select a random action uniformly from the action space.
        Returns:
            Tuple representing random action indices.
        """
        return tuple(random.randrange(dim) for dim in self.q_table.shape)

    def update(self, action: Tuple[int, ...], reward: float) -> None:
        """
        Update Q-value for a given action using incremental averaging.

        Args:
            action: Action index tuple.
            reward: Observed reward.
        """
        self.n_table[action] += 1
        n = self.n_table[action]
        #q_value = self.q_table[action]
        #self.q_table[action] = q_value + (reward - q_value) / float(n)
        self.q_table[action] += reward
        #self._decay_epsilon()

    def update(self, action: Tuple[int, ...], reward: float) -> None:
        """
        Update Q-value for a given action using incremental averaging.

        Args:
            action: Action index tuple.
            reward: Observed reward.
        """
        self.n_table[action] += 1
        n = self.n_table[action]
        
        q_value = self.q_table[action]
        
        self.q_table[action] = q_value + (reward - q_value) / float(n)
        
        # self._decay_epsilon()

    #def _decay_epsilon(self) -> None:
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
