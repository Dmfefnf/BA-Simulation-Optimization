"""Agents for the RL prototype of the lab-analysis simulation.

The action space is intentionally small and interpretable:

0 = FIFO
1 = Earliest Due Date
2 = Longest Waiting Time
3 = Highest Lateness Risk
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable

import numpy as np


class BaseAgent:
    """Common interface for all dispatching-rule agents."""

    def select_action(self, state: tuple[int, ...], training: bool = True) -> int:
        raise NotImplementedError

    def observe_transition(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        pass

    def start_episode(self) -> None:
        pass

    def end_episode(self) -> None:
        pass


class BaselineAgent(BaseAgent):
    """Agent that always returns one fixed dispatching rule."""

    def __init__(self, fixed_action: int = 0, n_actions: int = 4):
        if fixed_action not in range(n_actions):
            raise ValueError(f"fixed_action must be in 0..{n_actions - 1}")
        self.fixed_action = fixed_action
        self.n_actions = n_actions

    def select_action(self, state: tuple[int, ...], training: bool = True) -> int:
        return self.fixed_action


class RandomAgent(BaseAgent):
    """Uniform random dispatching-rule agent for smoke tests."""

    def __init__(self, n_actions: int = 4, random_seed: int | None = None):
        self.n_actions = n_actions
        self.rng = random.Random(random_seed)

    def select_action(self, state: tuple[int, ...], training: bool = True) -> int:
        return self.rng.randrange(self.n_actions)


class QLearningAgent(BaseAgent):
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        n_actions: int = 4,
        random_seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.rng = random.Random(random_seed)
        self.q_table: DefaultDict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def select_action(self, state: tuple[int, ...], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        q_values = self.q_table[state] if training else self.q_table.get(state)
        if q_values is None:
            return self.rng.randrange(self.n_actions)

        best_value = np.max(q_values)
        best_actions: Iterable[int] = np.flatnonzero(q_values == best_value)
        return int(self.rng.choice(list(best_actions)))

    def observe_transition(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        current = self.q_table[state][action]
        target = reward if done else reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] = current + self.alpha * (target - current)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "n_actions": self.n_actions,
            "q_table": dict(self.q_table),
        }
        with Path(path).open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            n_actions=payload["n_actions"],
        )
        agent.q_table.update(payload["q_table"])
        return agent
