import math

import numpy as np
import pytest

from rlfighter.core.action import ActionType, Phase
from rlfighter.core.agent import AgentState
from rlfighter.core.constants import BASE_HP, BASE_TOUGHNESS
from rlfighter.core.world import World


def test_phase_sequence_vertical():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    world.step({agent.agent_id: (ActionType.VERTICAL, 2)})  # face east
    assert agent.phase == Phase.WINDUP
    assert agent.phase_frame_remaining == 15
    for _ in range(15):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase == Phase.ACTIVE
    assert agent.phase_frame_remaining == 4
    for _ in range(4):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase == Phase.RECOVERY
    assert agent.phase_frame_remaining == 16
    for _ in range(16):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase == Phase.IDLE


def test_locked_during_animation():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    world.step({agent.agent_id: (ActionType.THRUST, 2)})
    # During windup, attempt to start another action
    for _ in range(6):
        world.step({agent.agent_id: (ActionType.VERTICAL, 2)})
        assert agent.action_type == ActionType.THRUST
    # Now in active
    assert agent.phase == Phase.ACTIVE
    for _ in range(4):
        world.step({agent.agent_id: (ActionType.VERTICAL, 2)})
        assert agent.action_type == ActionType.THRUST


def test_buffer_recovery():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    world.step({agent.agent_id: (ActionType.THRUST, 2)})
    # Advance to recovery (6 windup + 4 active = 10 frames)
    for _ in range(10):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase == Phase.RECOVERY
    assert agent.phase_frame_remaining == 8
    # Buffer too early should be ignored
    world.step({agent.agent_id: (ActionType.VERTICAL, 2)})
    assert agent.buffered_action is None
    # Advance to last 4 frames of recovery
    for _ in range(3):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase_frame_remaining == 4
    world.step({agent.agent_id: (ActionType.VERTICAL, 2)})
    assert agent.buffered_action == ActionType.VERTICAL
    # Finish recovery
    for _ in range(4):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase == Phase.WINDUP
    assert agent.action_type == ActionType.VERTICAL


def test_heal_mechanic():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    agent.hp = 30.0
    assert agent.heal_charges == 3
    world.step({agent.agent_id: (ActionType.HEAL, 8)})
    assert agent.phase == Phase.WINDUP
    assert agent.heal_charges == 3
    for _ in range(6):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
    # After windup ends, heal triggers and enters recovery
    assert agent.phase == Phase.RECOVERY
    assert agent.hp == pytest.approx(30.0 + BASE_HP * 0.5)
    assert agent.heal_charges == 2


def test_heal_no_charges():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    agent.heal_charges = 0
    agent.hp = 30.0
    world.step({agent.agent_id: (ActionType.HEAL, 8)})
    assert agent.phase == Phase.IDLE  # heal ignored
    assert agent.hp == 30.0


def test_heal_full_hp():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    assert agent.hp == BASE_HP
    world.step({agent.agent_id: (ActionType.HEAL, 8)})
    assert agent.phase == Phase.IDLE  # heal ignored
    assert agent.heal_charges == 3


def test_dodge_invincibility():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    world.step({agent.agent_id: (ActionType.DODGE, 8)})
    assert agent.phase == Phase.DODGE_INVINCIBLE
    assert agent.phase_frame_remaining == 10
    for _ in range(9):
        world.step({agent.agent_id: (ActionType.NOOP, 8)})
        assert agent.phase == Phase.DODGE_INVINCIBLE
    # One more step transitions to recovery
    world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.phase == Phase.RECOVERY


def test_movement_only_in_idle():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    start_pos = agent.pos.copy()
    world.step({agent.agent_id: (ActionType.NOOP, 0)})  # move north
    assert agent.phase == Phase.IDLE
    assert np.linalg.norm(agent.pos - start_pos) > 0.01

    start_pos = agent.pos.copy()
    world.step({agent.agent_id: (ActionType.VERTICAL, 0)})
    for _ in range(5):
        world.step({agent.agent_id: (ActionType.NOOP, 0)})
    assert agent.phase == Phase.WINDUP
    assert np.allclose(agent.pos, start_pos)


def test_death_clears_state():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    agent.hp = 0.0
    agent.alive = False
    world.step({agent.agent_id: (ActionType.VERTICAL, 2)})
    assert agent.phase == Phase.IDLE
    assert agent.action_type == ActionType.NOOP


def test_toughness_regen():
    world = World([1, 1], seed=0)
    agent = world.agents[0]
    agent.toughness = 50.0
    world.step({agent.agent_id: (ActionType.NOOP, 8)})
    assert agent.toughness > 50.0
