import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 96, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, 6)
        self.move_head = nn.Linear(hidden_dim, 9)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        x = self.trunk(obs)
        action_logits = self.action_head(x)
        move_logits = self.move_head(x)
        value = self.value_head(x)
        return action_logits, move_logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        action_logits, move_logits, value = self.forward(obs)
        action_dist = Categorical(logits=action_logits)
        move_dist = Categorical(logits=move_logits)

        if deterministic:
            action = action_dist.probs.argmax(dim=-1)
            move = move_dist.probs.argmax(dim=-1)
        else:
            action = action_dist.sample()
            move = move_dist.sample()

        log_prob = action_dist.log_prob(action) + move_dist.log_prob(move)
        return action, move, log_prob, value.squeeze(-1)
