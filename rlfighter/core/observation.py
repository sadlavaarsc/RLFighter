import math

import numpy as np

from rlfighter.core.action import FRAME_DATA, ActionType, Phase
from rlfighter.core.agent import AgentState
from rlfighter.core.constants import ARENA_SIZE, K_NEAREST
from rlfighter.core.world import World


def _one_hot(index: int, size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        arr[index] = 1.0
    return arr


def build_observation(agent: AgentState, world: World) -> np.ndarray:
    """Build fixed-length observation vector for an agent."""
    self_obs = _build_self_obs(agent)

    others = [a for a in world.agents if a.agent_id != agent.agent_id]
    others.sort(key=lambda a: float(np.linalg.norm(a.pos - agent.pos)))
    selected = others[:K_NEAREST]

    other_obs = []
    for i in range(K_NEAREST):
        if i < len(selected):
            other_obs.append(_build_other_obs(agent, selected[i]))
        else:
            other_obs.append(np.zeros(19, dtype=np.float32))

    return np.concatenate([self_obs] + other_obs)


def _phase_max_frames(action: ActionType, phase: Phase) -> int:
    """Return the total frame count for a given action+phase."""
    data = FRAME_DATA.get(action)
    if data is None:
        return 1
    if phase == Phase.WINDUP:
        return data.windup
    if phase == Phase.ACTIVE:
        return data.active
    if phase == Phase.RECOVERY:
        return data.recovery
    if phase == Phase.DODGE_INVINCIBLE:
        return data.active
    if phase == Phase.STAGGER:
        return 30
    return 1


def _build_self_obs(agent: AgentState) -> np.ndarray:
    hp_ratio = agent.hp / agent.max_hp if agent.max_hp > 0 else 0.0
    t_ratio = agent.toughness / agent.max_toughness if agent.max_toughness > 0 else 0.0
    heal_ratio = agent.heal_charges / 3.0

    phase_progress = 0.0
    max_frames = _phase_max_frames(agent.action_type, agent.phase)
    if max_frames > 0:
        phase_progress = 1.0 - (agent.phase_frame_remaining / max_frames)
        phase_progress = max(0.0, min(1.0, phase_progress))

    return np.concatenate([
        np.array([hp_ratio, t_ratio, heal_ratio], dtype=np.float32),
        _one_hot(agent.action_type.value, 6),
        _one_hot(agent.phase.value, 6),
        np.array([phase_progress, math.cos(agent.facing), math.sin(agent.facing),
                  agent.pos[0] / ARENA_SIZE, agent.pos[1] / ARENA_SIZE], dtype=np.float32),
    ])


def _build_other_obs(me: AgentState, other: AgentState) -> np.ndarray:
    diff = other.pos - me.pos
    dist = float(np.linalg.norm(diff))
    distance_norm = dist / (ARENA_SIZE * 1.414)

    rel_angle = math.atan2(diff[1], diff[0]) - me.facing
    rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))

    hp_ratio = other.hp / other.max_hp if other.max_hp > 0 else 0.0
    t_ratio = other.toughness / other.max_toughness if other.max_toughness > 0 else 0.0

    other_max_frames = _phase_max_frames(other.action_type, other.phase)
    other_phase_progress = 0.0
    if other_max_frames > 0:
        other_phase_progress = 1.0 - (other.phase_frame_remaining / other_max_frames)
        other_phase_progress = max(0.0, min(1.0, other_phase_progress))

    return np.concatenate([
        np.array([
            1.0,  # mask
            1.0 if other.team_id != me.team_id else 0.0,  # is_enemy
            1.0 if other.alive else 0.0,
            diff[0] / ARENA_SIZE,
            diff[1] / ARENA_SIZE,
            distance_norm,
            math.cos(rel_angle),
            math.sin(rel_angle),
        ], dtype=np.float32),
        _one_hot(other.action_type.value, 6),
        np.array([
            hp_ratio, t_ratio,
            math.cos(other.facing), math.sin(other.facing),
            other_phase_progress,
        ], dtype=np.float32),
    ])
