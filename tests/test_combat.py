import math

import numpy as np
import pytest

from rlfighter.core.action import ActionType, Phase
from rlfighter.core.agent import AgentState
from rlfighter.core.combat import resolve_combat
from rlfighter.core.constants import BASE_HP, BASE_TOUGHNESS
from rlfighter.core.world import World


def _make_agent(agent_id: int, team_id: int, x: float, y: float, facing: float) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        team_id=team_id,
        hp=BASE_HP,
        max_hp=BASE_HP,
        toughness=BASE_TOUGHNESS,
        max_toughness=BASE_TOUGHNESS,
        heal_charges=3,
        pos=np.array([x, y], dtype=np.float32),
        facing=facing,
    )


def test_vertical_hit():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)  # facing east
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.VERTICAL
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    resolve_combat([attacker, target])
    assert target.hp < BASE_HP
    assert target._hit_this_frame is True


def test_vertical_miss_out_of_range():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.VERTICAL
    target = _make_agent(1, 1, 5.0, 0.0, 0.0)
    resolve_combat([attacker, target])
    assert target.hp == BASE_HP


def test_horizontal_friendly_fire():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.HORIZONTAL
    ally = _make_agent(1, 0, 1.0, 0.0, 0.0)
    resolve_combat([attacker, ally])
    assert ally.hp < BASE_HP  # horizontal has friendly fire


def test_vertical_no_friendly_fire():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.VERTICAL
    ally = _make_agent(1, 0, 1.0, 0.0, 0.0)
    resolve_combat([attacker, ally])
    assert ally.hp == BASE_HP


def test_dodge_invincible():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.VERTICAL
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    target.phase = Phase.DODGE_INVINCIBLE
    resolve_combat([attacker, target])
    assert target.hp == BASE_HP


def test_toughness_break_stagger():
    world = World([1, 1], seed=0)
    attacker = world.agents[0]
    target = world.agents[1]
    # Position them close and facing
    attacker.pos = np.array([0.0, 0.0], dtype=np.float32)
    attacker.facing = 0.0
    target.pos = np.array([1.0, 0.0], dtype=np.float32)
    target.toughness = 10.0  # low toughness
    world.step({attacker.agent_id: (ActionType.VERTICAL, 2)})
    # Advance to active
    for _ in range(18):
        world.step({attacker.agent_id: (ActionType.NOOP, 8)})
    assert target.toughness == BASE_TOUGHNESS
    assert target.phase == Phase.STAGGER


def test_damage_kills():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.VERTICAL
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    target.hp = 10.0
    resolve_combat([attacker, target])
    assert target.alive is False
    assert target.phase == Phase.IDLE


def test_horizontal_arc_coverage():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)  # facing east
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.HORIZONTAL
    # Target at 45 degrees, within 110° arc and range
    target = _make_agent(1, 1, 0.7, 0.7, 0.0)
    resolve_combat([attacker, target])
    assert target.hp < BASE_HP


def test_horizontal_arc_miss():
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)  # facing east
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.HORIZONTAL
    # Target at 120 degrees, outside 110° arc
    angle = math.radians(120)
    target = _make_agent(1, 1, math.cos(angle), math.sin(angle), 0.0)
    resolve_combat([attacker, target])
    assert target.hp == BASE_HP
