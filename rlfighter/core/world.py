import math
import random
from typing import Optional

import numpy as np

from rlfighter.core.action import ActionType, FRAME_DATA, Phase, phase_sequence
from rlfighter.core.agent import AgentState
from rlfighter.core.combat import resolve_combat
from rlfighter.core.constants import (
    AGENT_RADIUS,
    AGENT_SPEED,
    ARENA_SIZE,
    BASE_HP,
    BASE_TOUGHNESS,
    FPS,
    HEAL_AMOUNT,
    MAX_HEAL_CHARGES,
    TOUGHNESS_REGEN_RATE,
)

_MOVE_DIRS = [
    np.array([0.0, 1.0], dtype=np.float32),   # N
    np.array([1.0, 1.0], dtype=np.float32),   # NE
    np.array([1.0, 0.0], dtype=np.float32),   # E
    np.array([1.0, -1.0], dtype=np.float32),  # SE
    np.array([0.0, -1.0], dtype=np.float32),  # S
    np.array([-1.0, -1.0], dtype=np.float32), # SW
    np.array([-1.0, 0.0], dtype=np.float32),  # W
    np.array([-1.0, 1.0], dtype=np.float32),  # NW
    np.array([0.0, 0.0], dtype=np.float32),   # stationary
]


def _move_vec(move_dir: int) -> np.ndarray:
    vec = _MOVE_DIRS[move_dir].copy()
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _phase_duration(action: ActionType, phase: Phase) -> int:
    data = FRAME_DATA[action]
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
    return 0


class World:
    def __init__(
        self,
        team_sizes: list[int],
        arena_size: float = ARENA_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        self.team_sizes = team_sizes
        self.arena_size = arena_size
        self.rng = random.Random(seed)
        self.agents: list[AgentState] = []
        self._next_id = 0
        self._init_agents()

    def _init_agents(self) -> None:
        self.agents.clear()
        self._next_id = 0
        margin = AGENT_RADIUS * 3.0
        for team_id, size in enumerate(self.team_sizes):
            for _ in range(size):
                pos = np.array([
                    self.rng.uniform(margin, self.arena_size - margin),
                    self.rng.uniform(margin, self.arena_size - margin),
                ], dtype=np.float32)
                facing = self.rng.uniform(-math.pi, math.pi)
                agent = AgentState(
                    agent_id=self._next_id,
                    team_id=team_id,
                    hp=BASE_HP,
                    max_hp=BASE_HP,
                    toughness=BASE_TOUGHNESS,
                    max_toughness=BASE_TOUGHNESS,
                    heal_charges=MAX_HEAL_CHARGES,
                    pos=pos,
                    facing=facing,
                )
                self.agents.append(agent)
                self._next_id += 1

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self._init_agents()

    def step(self, actions: dict[int, tuple[ActionType, int]]) -> None:
        """Advance one logic tick.

        actions maps agent_id -> (action_type, move_dir).
        """
        # 1. Phase transitions and new action starts
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.phase_frame_remaining > 0:
                agent.phase_frame_remaining -= 1

            if agent.phase_frame_remaining == 0:
                self._transition_phase(agent)

            # Accept new action only when IDLE or buffering in late RECOVERY
            action_type, move_dir = actions.get(agent.agent_id, (ActionType.NOOP, 8))

            if agent.phase == Phase.IDLE and action_type != ActionType.NOOP:
                self._start_action(agent, action_type, move_dir)
            elif (
                agent.phase == Phase.RECOVERY
                and agent.phase_frame_remaining <= 4
                and action_type != ActionType.NOOP
            ):
                agent.buffered_action = action_type
                agent.buffered_move_dir = move_dir

        # 2. Movement (only IDLE agents)
        dt = 1.0 / FPS
        for agent in self.agents:
            if not agent.alive or agent.phase != Phase.IDLE:
                continue
            _, move_dir = actions.get(agent.agent_id, (ActionType.NOOP, 8))
            if move_dir != 8:
                vec = _move_vec(move_dir)
                agent.pos += vec * AGENT_SPEED * dt
                agent.facing = math.atan2(vec[1], vec[0])
                self._clamp_pos(agent)

        # 3. Combat resolution
        resolve_combat(self.agents)

        # 4. Heal effect (agents that just finished HEAL windup)
        # This is handled inside _transition_phase: when HEAL windup -> recovery,
        # we trigger heal immediately.

        # 5. Toughness regeneration for agents not hit this frame
        for agent in self.agents:
            if not agent.alive:
                continue
            if not agent._hit_this_frame and agent.toughness < agent.max_toughness:
                agent.toughness += agent.max_toughness * TOUGHNESS_REGEN_RATE
                agent.toughness = min(agent.toughness, agent.max_toughness)

    def _transition_phase(self, agent: AgentState) -> None:
        if agent.phase == Phase.IDLE:
            return
        seq = phase_sequence(agent.action_type)
        try:
            idx = seq.index(agent.phase)
        except ValueError:
            idx = -1

        if idx >= 0 and idx + 1 < len(seq):
            next_phase = seq[idx + 1]
            agent.phase = next_phase
            agent.phase_frame_remaining = _phase_duration(agent.action_type, next_phase)
            # Heal triggers at the end of windup (transition to next phase)
            if agent.action_type == ActionType.HEAL and agent.phase == Phase.RECOVERY:
                if agent.heal_charges > 0 and agent.hp < agent.max_hp:
                    agent.heal_charges -= 1
                    agent.hp = min(agent.hp + agent.max_hp * HEAL_AMOUNT, agent.max_hp)
        else:
            # End of action sequence -> IDLE (or buffered action)
            if agent.buffered_action is not None:
                buf = agent.buffered_action
                mdir = agent.buffered_move_dir
                agent.buffered_action = None
                agent.buffered_move_dir = 8
                self._start_action(agent, buf, mdir)
            else:
                agent.phase = Phase.IDLE
                agent.action_type = ActionType.NOOP
                agent.phase_frame_remaining = 0

    def _start_action(self, agent: AgentState, action_type: ActionType, move_dir: int) -> None:
        if action_type == ActionType.HEAL and (agent.heal_charges <= 0 or agent.hp >= agent.max_hp):
            return
        agent.action_type = action_type
        seq = phase_sequence(action_type)
        if not seq:
            agent.phase = Phase.IDLE
            agent.phase_frame_remaining = 0
            return
        first_phase = seq[0]
        agent.phase = first_phase
        agent.phase_frame_remaining = _phase_duration(action_type, first_phase)
        # For movement attacks, face toward move_dir if provided
        if move_dir != 8 and action_type != ActionType.NOOP:
            vec = _move_vec(move_dir)
            agent.facing = math.atan2(vec[1], vec[0])

    def _clamp_pos(self, agent: AgentState) -> None:
        agent.pos[0] = max(AGENT_RADIUS, min(self.arena_size - AGENT_RADIUS, agent.pos[0]))
        agent.pos[1] = max(AGENT_RADIUS, min(self.arena_size - AGENT_RADIUS, agent.pos[1]))

    def alive_agents(self) -> list[AgentState]:
        return [a for a in self.agents if a.alive]

    def team_alive_count(self, team_id: int) -> int:
        return sum(1 for a in self.agents if a.team_id == team_id and a.alive)
