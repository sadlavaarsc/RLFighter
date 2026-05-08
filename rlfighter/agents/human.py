import pygame

from rlfighter.agents.base import Controller
from rlfighter.core.action import ActionType


class HumanController(Controller):
    """Keyboard controller using pygame keys."""

    def __init__(self) -> None:
        self.pressed = set()
        self.pending_action: ActionType | None = None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            self.pressed.add(event.key)
            if event.key == pygame.K_j:
                self.pending_action = ActionType.VERTICAL
            elif event.key == pygame.K_k:
                self.pending_action = ActionType.HORIZONTAL
            elif event.key == pygame.K_l:
                self.pending_action = ActionType.THRUST
            elif event.key == pygame.K_SPACE:
                self.pending_action = ActionType.DODGE
            elif event.key == pygame.K_h:
                self.pending_action = ActionType.HEAL
        elif event.type == pygame.KEYUP:
            self.pressed.discard(event.key)

    def act(self, agent_id: int, world) -> tuple[int, int]:
        action = self.pending_action
        self.pending_action = None
        if action is None:
            action = ActionType.NOOP

        move_dir = self._move_dir()
        return (action.value, move_dir)

    def _move_dir(self) -> int:
        w = pygame.K_w in self.pressed
        a = pygame.K_a in self.pressed
        s = pygame.K_s in self.pressed
        d = pygame.K_d in self.pressed

        if w and d:
            return 1
        if s and d:
            return 3
        if s and a:
            return 5
        if w and a:
            return 7
        if w:
            return 0
        if d:
            return 2
        if s:
            return 4
        if a:
            return 6
        return 8
