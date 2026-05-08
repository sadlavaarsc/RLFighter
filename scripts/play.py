import argparse
import sys

import pygame
import torch

from rlfighter.agents.base import Controller
from rlfighter.agents.human import HumanController
from rlfighter.agents.scripted import ScriptedController
from rlfighter.core.action import ActionType
from rlfighter.core.observation import build_observation
from rlfighter.core.world import World
from rlfighter.render.pygame_view import Renderer
from rlfighter.rl.policy import ActorCritic


def _resolve_controller(name: str, checkpoint: str | None = None) -> Controller:
    if name == "human":
        return HumanController()
    if name == "scripted":
        return ScriptedController()
    if name == "rl":
        if checkpoint is None:
            raise ValueError("RL controller requires --checkpoint")
        policy = ActorCritic(obs_dim=84, hidden_dim=128)
        ckpt = torch.load(checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        return _RLController(policy)
    raise ValueError(f"Unknown controller: {name}")


class _RLController(Controller):
    def __init__(self, policy: ActorCritic) -> None:
        self.policy = policy

    def act(self, agent_id: int, world) -> tuple[int, int]:
        for agent in world.agents:
            if agent.agent_id == agent_id:
                obs = build_observation(agent, world)
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    action_t, move_t, _, _ = self.policy.act(obs_t, deterministic=True)
                return (action_t.item(), move_t.item())
        return (0, 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="human", choices=["human", "scripted", "rl"])
    parser.add_argument("--p2", default="scripted", choices=["human", "scripted", "rl"])
    parser.add_argument("--checkpoint", help="checkpoint path for RL controller")
    parser.add_argument("--speed", type=int, default=30, help="render FPS")
    parser.add_argument("--max-ticks", type=int, default=3000)
    parser.add_argument("--auto-tick", action="store_true", help="no render delay")
    args = parser.parse_args()

    world = World(team_sizes=[1, 1], seed=0)
    renderer = Renderer()

    # Map each agent to a controller
    controllers: dict[int, Controller] = {}
    for agent in world.agents:
        if agent.team_id == 0:
            controllers[agent.agent_id] = _resolve_controller(args.p1, args.checkpoint)
        else:
            controllers[agent.agent_id] = _resolve_controller(args.p2, args.checkpoint)

    human = None
    for ctrl in controllers.values():
        if isinstance(ctrl, HumanController):
            human = ctrl
            break

    running = True
    tick = 0
    while running and tick < args.max_ticks:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if human:
                human.handle_event(event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break

        if not running:
            break

        # Collect actions
        actions = {}
        for agent in world.agents:
            if not agent.alive:
                continue
            ctrl = controllers[agent.agent_id]
            action_type_val, move_dir = ctrl.act(agent.agent_id, world)
            actions[agent.agent_id] = (ActionType(action_type_val), move_dir)

        world.step(actions)
        tick += 1

        renderer.draw(world, tick)
        if not args.auto_tick:
            renderer.tick_fps(args.speed)

        # Check win condition
        if world.team_alive_count(0) == 0 or world.team_alive_count(1) == 0:
            winner = 0 if world.team_alive_count(0) > 0 else 1
            print(f"Team {winner} wins at tick {tick}!")
            # Show final frame briefly
            if not args.auto_tick:
                pygame.time.wait(2000)
            break

    renderer.close()
    print(f"Game ended at tick {tick}")


if __name__ == "__main__":
    main()
