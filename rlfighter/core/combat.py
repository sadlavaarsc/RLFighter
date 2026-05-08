import math

import numpy as np

from rlfighter.core.action import ActionType, FRAME_DATA, IMPACT_LEVEL, Phase, POISE_LEVEL
from rlfighter.core.agent import AgentState
from rlfighter.core.constants import AGENT_RADIUS, BASE_HP, BASE_TOUGHNESS
from rlfighter.core.hitbox import point_in_oriented_rect, point_in_sector


def _in_hitbox(attacker: AgentState, target: AgentState) -> bool:
    """Check if target's center lies inside attacker's active hitbox."""
    data = FRAME_DATA[attacker.action_type]
    if attacker.action_type == ActionType.HORIZONTAL:
        return point_in_sector(
            target.pos,
            attacker.pos,
            attacker.facing,
            data.range,
            data.arc_deg,
        )
    elif attacker.action_type in (ActionType.VERTICAL, ActionType.THRUST):
        return point_in_oriented_rect(
            target.pos,
            attacker.pos,
            attacker.facing,
            data.width,
            data.length,
        )
    return False


def _break_action(agent: AgentState, long_stagger: bool = False) -> None:
    """Interrupt the agent's current action and put them into stagger."""
    agent.phase = Phase.STAGGER
    agent.phase_frame_remaining = 30 if long_stagger else 5
    agent.action_type = ActionType.NOOP
    agent.buffered_action = None
    agent.buffered_move_dir = 8
    agent._hit_targets.clear()


def resolve_combat(agents: list[AgentState]) -> None:
    """Apply damage, toughness depletion, stagger/interrupt, and mark hits.

    Each attacker can hit a given target at most once per action.
    Higher impact-level attacks can interrupt lower-poise actions.
    """
    for target in agents:
        if target.alive:
            target._hit_this_frame = False

    for attacker in agents:
        if not attacker.alive or attacker.phase != Phase.ACTIVE:
            continue
        data = FRAME_DATA[attacker.action_type]
        if data.damage <= 0.0:
            continue
        for target in agents:
            if target.agent_id == attacker.agent_id:
                continue
            if not target.alive:
                continue
            if target.team_id == attacker.team_id and not data.friendly_fire:
                continue
            if target.phase == Phase.DODGE_INVINCIBLE:
                continue
            if target.agent_id in attacker._hit_targets:
                continue
            if _in_hitbox(attacker, target):
                attacker._hit_targets.add(target.agent_id)
                target.hp -= data.damage
                target._hit_this_frame = True
                if target.hp <= 0.0:
                    target.hp = 0.0
                    target.alive = False
                    target.phase = Phase.IDLE
                    target.phase_frame_remaining = 0
                    target.buffered_action = None
                    target._hit_targets.clear()
                    continue
                target.toughness -= data.impact
                if target.toughness <= 0.0:
                    target.toughness = BASE_TOUGHNESS
                    _break_action(target, long_stagger=True)
                else:
                    attacker_level = IMPACT_LEVEL.get(attacker.action_type, 0)
                    target_poise = POISE_LEVEL.get(target.action_type, 0)
                    if attacker_level > target_poise:
                        _break_action(target, long_stagger=False)
                    elif target_poise == 0:
                        # Non-attack states (IDLE, RECOVERY, HEAL, etc.) always stagger
                        _break_action(target, long_stagger=False)
