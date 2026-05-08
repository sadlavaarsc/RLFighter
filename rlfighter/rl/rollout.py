import numpy as np


class RolloutBuffer:
    """Per-agent trajectory storage with GAE."""

    def __init__(self):
        self.observations: list[np.ndarray] = []
        self.actions: list[list[int]] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(self, obs: np.ndarray, action: list[int], log_prob: float, reward: float, value: float, done: bool) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95, last_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
