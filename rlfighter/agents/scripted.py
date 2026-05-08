import math

import numpy as np

from rlfighter.agents.base import Controller
from rlfighter.core.action import ActionType, FRAME_DATA, Phase
from rlfighter.core.world import World

_THREAT_RANGE = 2.5


def _dist(a, b) -> float:
    return float(np.linalg.norm(a.pos - b.pos))


def _angle_diff(a, b) -> float:
    diff = a - b
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


class ScriptedController(Controller):
    """Simple rule-based FSM agent.

    Tracks per-agent interaction history to avoid getting stuck in
    infinite dodge loops when no actual hits are landing.
    """

    # After this many frames without any hit landed or taken, skip dodge and push aggressively.
    RECKLESS_THRESHOLD = 90

    def __init__(self) -> None:
        # agent_id -> {other_id: hp} snapshot from previous act() call
        self._hp_snapshots: dict[int, dict[int, float]] = {}
        # agent_id -> frames since last interaction
        self._idle_frames: dict[int, int] = {}

    def act(self, agent_id: int, world: World) -> tuple[int, int]:
        me = world.agents[agent_id]
        if not me.alive:
            return (ActionType.NOOP.value, 8)

        # ---- interaction tracking (per-agent snapshot) ----
        prev_snapshot = self._hp_snapshots.get(agent_id, {})
        curr_snapshot = {a.agent_id: a.hp for a in world.agents}

        interaction = False
        if me.hp < prev_snapshot.get(agent_id, me.hp):
            interaction = True  # I got hit
        for other in world.agents:
            if other.team_id != me.team_id:
                if other.hp < prev_snapshot.get(other.agent_id, other.hp):
                    interaction = True  # I hit someone

        if interaction:
            self._idle_frames[agent_id] = 0
        else:
            self._idle_frames[agent_id] = self._idle_frames.get(agent_id, 0) + 1

        self._hp_snapshots[agent_id] = curr_snapshot
        reckless = self._idle_frames.get(agent_id, 0) > self.RECKLESS_THRESHOLD
        # ---------------------------------------------------

        enemies = [a for a in world.agents if a.team_id != me.team_id and a.alive]
        allies = [a for a in world.agents if a.team_id == me.team_id and a.agent_id != me.agent_id and a.alive]

        if not enemies:
            return (ActionType.NOOP.value, 8)

        target = min(enemies, key=lambda e: _dist(me, e))
        dist = _dist(me, target)
        angle_to_target = math.atan2(target.pos[1] - me.pos[1], target.pos[0] - me.pos[0])
        face_diff = abs(_angle_diff(angle_to_target, me.facing))

        # 1. Heal if safe and low HP
        if me.hp < me.max_hp * 0.4 and me.heal_charges > 0:
            safe = all(_dist(me, e) > _THREAT_RANGE for e in enemies)
            if safe:
                return (ActionType.HEAL.value, 8)

        # 2. Dodge if enemy is winding up and the attack can actually reach us.
        #    When "reckless" (stuck for too long) we skip this entirely.
        if not reckless and target.phase == Phase.WINDUP and dist < _THREAT_RANGE:
            attack_data = FRAME_DATA.get(target.action_type)
            if attack_data is not None and attack_data.damage > 0.0:
                can_hit = False
                if target.action_type == ActionType.HORIZONTAL:
                    can_hit = dist <= attack_data.range + 0.2
                elif target.action_type in (ActionType.VERTICAL, ActionType.THRUST):
                    can_hit = dist <= attack_data.length + 0.3
                if can_hit:
                    return (ActionType.DODGE.value, 8)

        # 0. Face the target before committing to an attack.
        #    When idle we can snap to the exact angle in-place (no movement needed).
        max_attack_range = max(
            FRAME_DATA[ActionType.VERTICAL].length,
            FRAME_DATA[ActionType.THRUST].length,
            FRAME_DATA[ActionType.HORIZONTAL].range,
        )
        if me.phase == Phase.IDLE and dist <= max_attack_range and face_diff > math.radians(30):
            me.facing = angle_to_target
            return (ActionType.NOOP.value, 8)

        # 3. Thrust if close and facing
        thrust_data = FRAME_DATA[ActionType.THRUST]
        if dist <= thrust_data.length and face_diff < math.radians(15):
            return (ActionType.THRUST.value, 8)

        # 4. Horizontal if multiple enemies clustered and no ally in arc
        if len(enemies) >= 2:
            nearby_enemies = [e for e in enemies if _dist(me, e) <= FRAME_DATA[ActionType.HORIZONTAL].range]
            if len(nearby_enemies) >= 2:
                arc_rad = math.radians(FRAME_DATA[ActionType.HORIZONTAL].arc_deg)
                ally_in_arc = False
                for ally in allies:
                    a_angle = math.atan2(ally.pos[1] - me.pos[1], ally.pos[0] - me.pos[0])
                    if _dist(me, ally) <= FRAME_DATA[ActionType.HORIZONTAL].range and abs(_angle_diff(a_angle, me.facing)) < arc_rad / 2:
                        ally_in_arc = True
                        break
                if not ally_in_arc:
                    return (ActionType.HORIZONTAL.value, 8)

        # 5. Vertical if in range
        vert_data = FRAME_DATA[ActionType.VERTICAL]
        if dist <= vert_data.length and face_diff < math.radians(30):
            return (ActionType.VERTICAL.value, 8)

        # 6. Move toward target (only movement uses 8-way discrete dirs)
        return (ActionType.NOOP.value, self._move_dir(me, target))

    def _move_dir(self, me, target) -> int:
        dx = target.pos[0] - me.pos[0]
        dy = target.pos[1] - me.pos[1]
        angle = math.atan2(dy, dx)

        # Map angle to 8 directions
        # 0=N(pi/2), 1=NE(pi/4), 2=E(0), 3=SE(-pi/4), 4=S(-pi/2), 5=SW(-3pi/4), 6=W(pi), 7=NW(3pi/4)
        dirs = [
            (0, math.pi / 2),       # N
            (1, math.pi / 4),       # NE
            (2, 0.0),               # E
            (3, -math.pi / 4),      # SE
            (4, -math.pi / 2),      # S
            (5, -3 * math.pi / 4),  # SW
            (6, math.pi),           # W
            (7, 3 * math.pi / 4),   # NW
        ]
        best_dir = 2
        best_diff = float('inf')
        for d, target_angle in dirs:
            diff = abs(_angle_diff(angle, target_angle))
            if diff < best_diff:
                best_diff = diff
                best_dir = d
        return best_dir
