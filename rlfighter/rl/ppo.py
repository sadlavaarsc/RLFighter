import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from rlfighter.rl.rollout import RolloutBuffer


def ppo_update(
    policy,
    optimizer: torch.optim.Optimizer,
    buffers: list[RolloutBuffer],
    epochs: int = 4,
    batch_size: int = 256,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_kl: float = 0.03,
) -> dict:
    """Run PPO update on collected rollouts."""
    all_obs: list[torch.Tensor] = []
    all_actions: list[list[int]] = []
    all_old_log_probs: list[float] = []
    all_returns: list[float] = []
    all_advantages: list[float] = []

    for buf in buffers:
        if len(buf.observations) == 0:
            continue
        returns, advantages = buf.compute_advantages(last_value=0.0)
        for i in range(len(buf.observations)):
            all_obs.append(torch.from_numpy(buf.observations[i]))
            all_actions.append(buf.actions[i])
            all_old_log_probs.append(buf.log_probs[i])
            all_returns.append(float(returns[i]))
            all_advantages.append(float(advantages[i]))

    if len(all_obs) == 0:
        return {}

    obs_tensor = torch.stack(all_obs)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)
    old_log_probs_tensor = torch.tensor(all_old_log_probs, dtype=torch.float32)
    returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
    advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)

    # Normalize advantages
    adv_mean = advantages_tensor.mean()
    adv_std = advantages_tensor.std()
    advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)

    dataset = TensorDataset(obs_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    num_batches = 0
    kl_exceeded = False

    for epoch in range(epochs):
        for batch in loader:
            obs_b, actions_b, old_log_probs_b, returns_b, advantages_b = batch

            action_logits, move_logits, values = policy(obs_b)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            move_dist = torch.distributions.Categorical(logits=move_logits)

            action = actions_b[:, 0]
            move = actions_b[:, 1]

            log_prob = action_dist.log_prob(action) + move_dist.log_prob(move)
            ratio = torch.exp(log_prob - old_log_probs_b)

            surr1 = ratio * advantages_b
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_b
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns_b)

            entropy = action_dist.entropy().mean() + move_dist.entropy().mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                kl = (old_log_probs_b - log_prob).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl
            num_batches += 1

            if kl > max_kl:
                kl_exceeded = True
                break
        if kl_exceeded:
            break

    return {
        "policy_loss": total_policy_loss / max(num_batches, 1),
        "value_loss": total_value_loss / max(num_batches, 1),
        "entropy": total_entropy / max(num_batches, 1),
        "kl": total_kl / max(num_batches, 1),
        "num_batches": num_batches,
    }
