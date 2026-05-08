from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    NOOP = 0
    VERTICAL = 1
    HORIZONTAL = 2
    THRUST = 3
    DODGE = 4
    HEAL = 5


class Phase(Enum):
    IDLE = 0
    WINDUP = 1
    ACTIVE = 2
    RECOVERY = 3
    STAGGER = 4
    DODGE_INVINCIBLE = 5


@dataclass(frozen=True)
class FrameData:
    windup: int
    active: int
    recovery: int
    damage: float = 0.0
    impact: float = 0.0
    range: float = 0.0
    arc_deg: float = 0.0
    width: float = 0.0
    length: float = 0.0
    friendly_fire: bool = False


FRAME_DATA: dict[ActionType, FrameData] = {
    ActionType.VERTICAL: FrameData(
        windup=18,
        active=4,
        recovery=14,
        damage=25.0,
        impact=35.0,
        width=0.4,
        length=1.6,
        friendly_fire=False,
    ),
    ActionType.HORIZONTAL: FrameData(
        windup=10,
        active=8,
        recovery=12,
        damage=18.0,
        impact=25.0,
        range=1.4,
        arc_deg=110.0,
        friendly_fire=True,
    ),
    ActionType.THRUST: FrameData(
        windup=8,
        active=6,
        recovery=10,
        damage=20.0,
        impact=30.0,
        width=0.3,
        length=2.0,
        friendly_fire=False,
    ),
    ActionType.DODGE: FrameData(
        windup=0,
        active=10,
        recovery=8,
    ),
    ActionType.HEAL: FrameData(
        windup=6,
        active=0,
        recovery=18,
    ),
}


def phase_sequence(action: ActionType) -> list[Phase]:
    data = FRAME_DATA[action]
    seq = []
    if data.windup > 0:
        seq.append(Phase.WINDUP)
    if action == ActionType.DODGE:
        seq.append(Phase.DODGE_INVINCIBLE)
    elif data.active > 0:
        seq.append(Phase.ACTIVE)
    seq.append(Phase.RECOVERY)
    return seq
