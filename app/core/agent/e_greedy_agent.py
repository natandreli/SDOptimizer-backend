import random
from typing import Tuple

import numpy as np


class EGreedyAgent:
    """
    ε-greedy multi-armed bandit agent.

    Maintains action-value estimates (Q-values) and updates them using
    incremental averaging based on observed rewards.
    """

    def __init__(self, action_shape: Tuple[int, ...], epsilon: float) -> None:
        """
        Initialize the agent.

        Args:
            action_shape: Shape of the action space.
                Example for v parameters: (3, 3, ..., 3)
            epsilon: Exploration probability in [0, 1].

        Raises:
            ValueError: If epsilon is outside valid range.
        """
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        self.epsilon = epsilon
        self.q_table = np.zeros(action_shape)
        self.n_table = np.zeros(action_shape, dtype=int)

    def select_action(self) -> Tuple[int, ...]:
        """
        Select an action using ε-greedy policy.

        Returns:
            Tuple representing selected action indices.
        """
        if random.random() > self.epsilon:
            max_val = np.max(self.q_table)
            candidates = np.argwhere(self.q_table == max_val)
            idx = candidates[random.randint(0, len(candidates) - 1)]
            return tuple(idx)
        else:
            return tuple(random.randint(0, dim - 1) for dim in self.q_table.shape)

    def update(self, action: Tuple[int, ...], reward: float) -> None:
        """
        Update Q-value for a given action using incremental averaging.

        Args:
            action: Action index tuple.
            reward: Observed reward.
        """
        self.n_table[action] += 1
        n = self.n_table[action]
        self.q_table[action] += (1.0 / n) * (reward - self.q_table[action])
