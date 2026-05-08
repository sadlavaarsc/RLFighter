import argparse

import numpy as np
import torch

from rlfighter.agents.scripted import ScriptedController
from rlfighter.core.action import ActionType
from rlfighter.core.observation import build_observation
from rlfighter.env.arena import Arena
from rlfighter.rl.policy import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teams", default="1,1")
    parser.add_argument("--opponent", default="scripted", choices=["scripted", "self"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    team_sizes = [int(x) for x in args.teams.split(",")]
    device = torch.device("cpu")

    policy = ActorCritic(obs_dim=84, hidden_dim=128).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    scripted = ScriptedController()
    env = Arena(team_sizes, max_steps=args.max_steps, seed=args.seed)

    wins = 0
    ep_lengths = []
    ep_rewards = []

    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < args.max_steps:
            actions_dict = {}
            for agent in env.world.agents:
                if not agent.alive:
                    continue
                if agent.team_id == 0:
                    ob = build_observation(agent, env.world)
                    ob_t = torch.from_numpy(ob).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        action_t, move_t, _, _ = policy.act(ob_t, deterministic=True)
                    actions_dict[agent.agent_id] = (ActionType(action_t.item()), move_t.item())
                else:
                    if args.opponent == "scripted":
                        a, m = scripted.act(agent.agent_id, env.world)
                    else:
                        ob = build_observation(agent, env.world)
                        ob_t = torch.from_numpy(ob).float().unsqueeze(0).to(device)
                        with torch.no_grad():
                            action_t, move_t, _, _ = policy.act(ob_t, deterministic=True)
                        a = action_t.item()
                        m = move_t.item()
                    actions_dict[agent.agent_id] = (ActionType(a), m)

            obs, rewards, terminated, truncated, info = env.step(actions_dict)
            step += 1
            ep_reward += sum(rewards.values()) / max(len(rewards), 1)
            done = info.get("winner") is not None or all(terminated.values()) or all(truncated.values())

        winner = info.get("winner")
        if winner == 0:
            wins += 1
        ep_lengths.append(step)
        ep_rewards.append(ep_reward)

    print(f"Episodes: {args.episodes}")
    print(f"Win rate (team 0): {wins / args.episodes:.2%}")
    print(f"Avg episode length: {np.mean(ep_lengths):.1f}")
    print(f"Avg episode reward: {np.mean(ep_rewards):.3f}")


if __name__ == "__main__":
    main()
