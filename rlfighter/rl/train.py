import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rlfighter.agents.scripted import ScriptedController
from rlfighter.core.action import ActionType
from rlfighter.core.observation import build_observation
from rlfighter.env.arena import Arena
from rlfighter.rl.policy import ActorCritic
from rlfighter.rl.ppo import ppo_update
from rlfighter.rl.rollout import RolloutBuffer


class Trainer:
    def __init__(
        self,
        team_sizes: list[int],
        opponent: str = "scripted",
        num_envs: int = 4,
        steps_per_update: int = 2048,
        total_steps: int = 1_000_000,
        lr: float = 3e-4,
        hidden_dim: int = 128,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epochs: int = 4,
        batch_size: int = 256,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_kl: float = 0.03,
        seed: int = 0,
        save_interval: int = 10,
        logdir: str = "runs",
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
    ) -> None:
        self.team_sizes = team_sizes
        self.opponent_type = opponent
        self.num_envs = num_envs
        self.steps_per_update = steps_per_update
        self.total_steps = total_steps
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_kl = max_kl
        self.seed = seed
        self.save_interval = save_interval
        self.logdir = logdir
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy = ActorCritic(obs_dim=84, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.envs = [Arena(team_sizes, seed=seed + i) for i in range(num_envs)]
        for env in self.envs:
            env.reset()

        self.scripted = ScriptedController() if opponent == "scripted" else None

        self.writer = SummaryWriter(log_dir=f"{logdir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Episode tracking
        self._ep_returns: dict[tuple[int, int], float] = {}
        self._ep_lengths: dict[tuple[int, int], int] = {}

    def collect_rollouts(self) -> list[RolloutBuffer]:
        """Collect one update worth of transitions from all envs."""
        buffers: dict[tuple[int, int], RolloutBuffer] = {}
        for env_idx, env in enumerate(self.envs):
            for agent in env.world.agents:
                key = (env_idx, agent.agent_id)
                if key not in buffers:
                    buffers[key] = RolloutBuffer()

        step_count = 0
        while step_count < self.steps_per_update:
            for env_idx, env in enumerate(self.envs):
                actions_dict = {}
                for agent in env.world.agents:
                    if not agent.alive:
                        continue
                    obs = build_observation(agent, env.world)
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                    if agent.team_id == 0 or self.opponent_type == "self_play":
                        with torch.no_grad():
                            action_t, move_t, log_prob_t, value_t = self.policy.act(obs_tensor)
                        action = action_t.item()
                        move = move_t.item()
                        log_prob = log_prob_t.item()
                        value = value_t.item()
                    else:
                        action_val, move = self.scripted.act(agent.agent_id, env.world)
                        action = action_val
                        log_prob = 0.0
                        value = 0.0

                    actions_dict[agent.agent_id] = (ActionType(action), move)
                    key = (env_idx, agent.agent_id)
                    buf = buffers[key]
                    buf.observations.append(obs)
                    buf.actions.append([action, move])
                    buf.log_probs.append(log_prob)
                    buf.values.append(value)

                obs_next, rewards, terminated, truncated, info = env.step(actions_dict)

                for agent in env.world.agents:
                    key = (env_idx, agent.agent_id)
                    if key not in buffers:
                        continue
                    buf = buffers[key]
                    if len(buf.observations) > len(buf.rewards):
                        r = rewards.get(agent.agent_id, 0.0)
                        done = terminated.get(agent.agent_id, False) or truncated.get(agent.agent_id, False)
                        buf.rewards.append(r)
                        buf.dones.append(done)

                        # Episode tracking
                        self._ep_returns[key] = self._ep_returns.get(key, 0.0) + r
                        self._ep_lengths[key] = self._ep_lengths.get(key, 0) + 1
                        if done:
                            self.writer.add_scalar(f"ep_return/env{env_idx}_agent{agent.agent_id}", self._ep_returns[key], self._total_step)
                            self.writer.add_scalar(f"ep_length/env{env_idx}_agent{agent.agent_id}", self._ep_lengths[key], self._total_step)
                            self._ep_returns[key] = 0.0
                            self._ep_lengths[key] = 0

                if info.get("winner") is not None or all(terminated.values()) or all(truncated.values()):
                    env.reset()

            step_count += 1
            self._total_step += 1

        return list(buffers.values())

    def train(self) -> None:
        num_updates = self.total_steps // self.steps_per_update
        self._total_step = 0

        for update in range(num_updates):
            buffers = self.collect_rollouts()

            metrics = ppo_update(
                self.policy,
                self.optimizer,
                buffers,
                epochs=self.epochs,
                batch_size=self.batch_size,
                clip_eps=self.clip_eps,
                vf_coef=self.vf_coef,
                ent_coef=self.ent_coef,
                max_kl=self.max_kl,
            )

            for buf in buffers:
                buf.clear()

            if metrics:
                self.writer.add_scalar("train/policy_loss", metrics["policy_loss"], self._total_step)
                self.writer.add_scalar("train/value_loss", metrics["value_loss"], self._total_step)
                self.writer.add_scalar("train/entropy", metrics["entropy"], self._total_step)
                self.writer.add_scalar("train/kl", metrics["kl"], self._total_step)

            print(
                f"Update {update + 1}/{num_updates} | Step {self._total_step} | "
                f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                f"Value Loss: {metrics.get('value_loss', 0):.4f} | "
                f"Entropy: {metrics.get('entropy', 0):.4f} | "
                f"KL: {metrics.get('kl', 0):.4f}"
            )

            if (update + 1) % self.save_interval == 0:
                path = os.path.join(self.checkpoint_dir, f"checkpoint_{self._total_step}.pt")
                torch.save(
                    {
                        "policy": self.policy.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "step": self._total_step,
                    },
                    path,
                )
                print(f"Saved checkpoint to {path}")

        final_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self._total_step,
            },
            final_path,
        )
        print(f"Training complete. Final checkpoint: {final_path}")
        self.writer.close()
