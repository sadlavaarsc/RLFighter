from dataclasses import dataclass, field
import numpy as np

from rlfighter.core.action import ActionType, Phase


@dataclass
class AgentState:
    agent_id: int
    team_id: int

    hp: float
    max_hp: float
    toughness: float
    max_toughness: float
    heal_charges: int

    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    facing: float = 0.0

    action_type: ActionType = ActionType.NOOP
    phase: Phase = Phase.IDLE
    phase_frame_remaining: int = 0
    buffered_action: ActionType | None = None
    buffered_move_dir: int = 8

    alive: bool = True
    _hit_this_frame: bool = False
    _hit_targets: set[int] = field(default_factory=set)

    def reset(self, pos: np.ndarray, facing: float = 0.0) -> None:
        self.hp = self.max_hp
        self.toughness = self.max_toughness
        self.heal_charges = 3
        self.pos = pos.copy()
        self.facing = facing
        self.action_type = ActionType.NOOP
        self.phase = Phase.IDLE
        self.phase_frame_remaining = 0
        self.buffered_action = None
        self.buffered_move_dir = 8
        self.alive = True
        self._hit_this_frame = False
        self._hit_targets.clear()
