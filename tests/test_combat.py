import math

import numpy as np
import pytest

from rlfighter.core.action import ActionType, FRAME_DATA, Phase
from rlfighter.core.agent import AgentState
from rlfighter.core.combat import resolve_combat
from rlfighter.core.constants import BASE_HP, BASE_TOUGHNESS
from rlfighter.core.world import World
from rlfighter.env.arena import Arena


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
    for _ in range(15):
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


def test_single_hit_per_action():
    """Multi-frame ACTIVE attacks should only hit a target once per action."""
    world = World([1, 1], seed=0)
    attacker = world.agents[0]
    target = world.agents[1]
    attacker.pos = np.array([0.0, 0.0], dtype=np.float32)
    attacker.facing = 0.0
    target.pos = np.array([1.0, 0.0], dtype=np.float32)
    target.hp = BASE_HP

    # Start a horizontal attack (active=6 frames)
    world.step({attacker.agent_id: (ActionType.HORIZONTAL, 2)})
    # Advance through active and recovery (10 windup + 6 active + 10 recovery)
    for _ in range(26):
        world.step({attacker.agent_id: (ActionType.NOOP, 8)})
    assert attacker.phase == Phase.IDLE

    # HP should only drop by one hit's worth, not 6×
    assert target.hp == BASE_HP - FRAME_DATA[ActionType.HORIZONTAL].damage
    assert target.agent_id in attacker._hit_targets


def test_miss_penalty():
    """Swinging and missing should incur a penalty when ACTIVE ends."""
    arena = Arena([1, 1], seed=0)
    attacker = arena.world.agents[0]
    target = arena.world.agents[1]
    # Place target far away so attack will miss
    attacker.pos = np.array([0.0, 0.0], dtype=np.float32)
    attacker.facing = 0.0
    target.pos = np.array([10.0, 0.0], dtype=np.float32)

    # Start a vertical attack (slow, heavy miss penalty)
    arena.step({attacker.agent_id: (ActionType.VERTICAL, 2)})
    # Advance through windup (15) + active (4); miss penalty fires on the
    # ACTIVE→RECOVERY transition frame.
    transition_reward = None
    for _ in range(19):
        _, r, _, _, _ = arena.step({attacker.agent_id: (ActionType.NOOP, 8)})
        if attacker.phase == Phase.RECOVERY and transition_reward is None:
            transition_reward = r[attacker.agent_id]

    assert attacker.phase == Phase.RECOVERY
    assert attacker._hit_targets == set()
    # In 1v1 team reward scales individual by 1.3x; miss penalty = -10.001 * 1.3
    assert transition_reward == pytest.approx(-13.0013, abs=0.1)


def test_miss_penalty_scale():
    """Miss penalty: THRUST < HORIZONTAL < VERTICAL."""
    penalties = {}
    for action in (ActionType.THRUST, ActionType.HORIZONTAL, ActionType.VERTICAL):
        arena = Arena([1, 1], seed=0)
        attacker = arena.world.agents[0]
        target = arena.world.agents[1]
        attacker.pos = np.array([0.0, 0.0], dtype=np.float32)
        attacker.facing = 0.0
        target.pos = np.array([10.0, 0.0], dtype=np.float32)

        arena.step({attacker.agent_id: (action, 2)})
        # Advance through windup + active; capture the ACTIVE→RECOVERY reward
        windup = FRAME_DATA[action].windup
        active = FRAME_DATA[action].active
        transition_reward = None
        for _ in range(windup + active):
            _, r, _, _, _ = arena.step({attacker.agent_id: (ActionType.NOOP, 8)})
            if attacker.phase == Phase.RECOVERY and transition_reward is None:
                transition_reward = r[attacker.agent_id]

        penalties[action] = transition_reward

    assert penalties[ActionType.THRUST] > penalties[ActionType.HORIZONTAL]
    assert penalties[ActionType.HORIZONTAL] > penalties[ActionType.VERTICAL]


def test_no_miss_penalty_on_hit():
    """Hitting the target should not trigger miss penalty."""
    arena = Arena([1, 1], seed=0)
    attacker = arena.world.agents[0]
    target = arena.world.agents[1]
    attacker.pos = np.array([0.0, 0.0], dtype=np.float32)
    attacker.facing = 0.0
    target.pos = np.array([1.0, 0.0], dtype=np.float32)

    arena.step({attacker.agent_id: (ActionType.VERTICAL, 2)})
    # Advance through windup (15) + active (4)
    first_active_reward = None
    transition_reward = None
    for _ in range(19):
        _, r, _, _, _ = arena.step({attacker.agent_id: (ActionType.NOOP, 8)})
        if attacker.phase == Phase.ACTIVE and first_active_reward is None:
            first_active_reward = r[attacker.agent_id]
        if attacker.phase == Phase.RECOVERY and transition_reward is None:
            transition_reward = r[attacker.agent_id]

    # Damage reward applies on the first active frame (~+45.5 with team scaling)
    assert first_active_reward == pytest.approx(45.4987, abs=0.1)
    # On the transition frame there is no miss penalty, just step penalty (~-0.0013)
    assert transition_reward == pytest.approx(-0.0013, abs=0.01)


def test_high_level_interrupts_low_level():
    """VERTICAL (level 3) should interrupt THRUST (level 1)."""
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.VERTICAL
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    target.phase = Phase.ACTIVE
    target.action_type = ActionType.THRUST
    resolve_combat([attacker, target])
    assert target.phase == Phase.STAGGER
    assert target.action_type == ActionType.NOOP


def test_low_level_cannot_interrupt_high_level():
    """THRUST (level 1) should not interrupt VERTICAL (level 3)."""
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.THRUST
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    target.phase = Phase.ACTIVE
    target.action_type = ActionType.VERTICAL
    resolve_combat([attacker, target])
    assert target.phase == Phase.ACTIVE
    assert target.action_type == ActionType.VERTICAL


def test_equal_level_no_interrupt():
    """Same-level attacks trade hits without interrupting each other."""
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.HORIZONTAL
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    target.phase = Phase.ACTIVE
    target.action_type = ActionType.HORIZONTAL
    resolve_combat([attacker, target])
    assert target.phase == Phase.ACTIVE
    assert target.action_type == ActionType.HORIZONTAL


def test_non_attack_state_still_staggers():
    """IDLE and RECOVERY targets always get staggered on hit."""
    for phase in (Phase.IDLE, Phase.RECOVERY):
        attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
        attacker.phase = Phase.ACTIVE
        attacker.action_type = ActionType.THRUST
        target = _make_agent(1, 1, 1.0, 0.0, 0.0)
        target.phase = phase
        target.action_type = ActionType.NOOP
        resolve_combat([attacker, target])
        assert target.phase == Phase.STAGGER, f"failed for phase {phase}"


def test_interrupt_clears_buffered_action():
    """Getting interrupted should discard any buffered follow-up."""
    attacker = _make_agent(0, 0, 0.0, 0.0, 0.0)
    attacker.phase = Phase.ACTIVE
    attacker.action_type = ActionType.HORIZONTAL
    target = _make_agent(1, 1, 1.0, 0.0, 0.0)
    target.phase = Phase.ACTIVE
    target.action_type = ActionType.THRUST
    target.buffered_action = ActionType.VERTICAL
    target.buffered_move_dir = 2
    resolve_combat([attacker, target])
    assert target.phase == Phase.STAGGER
    assert target.buffered_action is None
    assert target.buffered_move_dir == 8
