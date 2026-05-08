import math

import numpy as np
import pygame

from rlfighter.core.action import ActionType, FRAME_DATA, Phase
from rlfighter.core.constants import AGENT_RADIUS, ARENA_SIZE
from rlfighter.core.hitbox import _facing_vector
from rlfighter.core.world import World

_SCALE = 40.0
_WINDOW_SIZE = int(ARENA_SIZE * _SCALE)

_PHASE_COLORS = {
    Phase.IDLE: (220, 220, 220),
    Phase.WINDUP: (255, 220, 0),
    Phase.ACTIVE: (255, 60, 60),
    Phase.RECOVERY: (60, 120, 255),
    Phase.STAGGER: (120, 120, 120),
    Phase.DODGE_INVINCIBLE: (0, 255, 255),
}

_TEAM_COLORS = {
    0: (60, 120, 220),
    1: (220, 60, 60),
    2: (60, 200, 60),
    3: (200, 60, 200),
}


def _world_to_screen(pos: np.ndarray) -> tuple[int, int]:
    x = int(pos[0] * _SCALE)
    y = int(_WINDOW_SIZE - pos[1] * _SCALE)
    return (x, y)


def _draw_hp_bar(surface: pygame.Surface, center: tuple[int, int], hp: float, max_hp: float) -> None:
    w = 32
    h = 4
    x = center[0] - w // 2
    y = center[1] - 28
    ratio = hp / max_hp
    pygame.draw.rect(surface, (40, 40, 40), (x, y, w, h))
    pygame.draw.rect(surface, (60, 220, 60), (x, y, int(w * ratio), h))


def _draw_toughness_bar(surface: pygame.Surface, center: tuple[int, int], t: float, max_t: float) -> None:
    w = 32
    h = 3
    x = center[0] - w // 2
    y = center[1] - 22
    ratio = t / max_t
    pygame.draw.rect(surface, (40, 40, 40), (x, y, w, h))
    pygame.draw.rect(surface, (255, 160, 40), (x, y, int(w * ratio), h))


def _draw_hitbox(surface: pygame.Surface, agent) -> None:
    if agent.phase != Phase.ACTIVE:
        return
    data = FRAME_DATA[agent.action_type]
    color = (255, 80, 80, 80)
    center = _world_to_screen(agent.pos)

    if agent.action_type == ActionType.HORIZONTAL:
        radius = int(data.range * _SCALE)
        arc = math.radians(data.arc_deg)
        fv = _facing_vector(agent.facing)
        base_angle = math.atan2(-fv[1], fv[0])  # screen y is inverted
        points = [center]
        steps = 20
        for i in range(steps + 1):
            theta = base_angle - arc / 2 + arc * i / steps
            px = center[0] + radius * math.cos(theta)
            py = center[1] + radius * math.sin(theta)
            points.append((int(px), int(py)))
        points.append(center)
        s = pygame.Surface((_WINDOW_SIZE, _WINDOW_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(s, color, points)
        surface.blit(s, (0, 0))
    elif agent.action_type in (ActionType.VERTICAL, ActionType.THRUST):
        # Draw oriented rectangle as a thin polygon
        fv = _facing_vector(agent.facing)
        pv = np.array([-fv[1], fv[0]], dtype=np.float32)
        length = data.length
        width = data.width
        p0 = agent.pos
        p1 = p0 + fv * length + pv * (width / 2)
        p2 = p0 + fv * length - pv * (width / 2)
        points = [_world_to_screen(p0), _world_to_screen(p1), _world_to_screen(p2)]
        s = pygame.Surface((_WINDOW_SIZE, _WINDOW_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(s, color, points)
        surface.blit(s, (0, 0))


def _draw_agent(surface: pygame.Surface, agent) -> None:
    if not agent.alive:
        return
    center = _world_to_screen(agent.pos)
    radius = int(AGENT_RADIUS * _SCALE)
    phase_color = _PHASE_COLORS.get(agent.phase, (200, 200, 200))
    team_color = _TEAM_COLORS.get(agent.team_id, (200, 200, 200))

    # Outer ring = phase, inner = team
    pygame.draw.circle(surface, phase_color, center, radius + 4)
    pygame.draw.circle(surface, team_color, center, radius)

    # Facing arrow
    angle = agent.facing
    ax = center[0] + (radius + 8) * math.cos(angle)
    ay = center[1] - (radius + 8) * math.sin(angle)
    pygame.draw.line(surface, (255, 255, 255), center, (int(ax), int(ay)), 2)

    # HP / toughness
    _draw_hp_bar(surface, center, agent.hp, agent.max_hp)
    _draw_toughness_bar(surface, center, agent.toughness, agent.max_toughness)

    # Heal charges
    for i in range(agent.heal_charges):
        cx = center[0] - 8 + i * 6
        cy = center[1] + radius + 6
        pygame.draw.circle(surface, (60, 255, 60), (cx, cy), 2)

    # Action label
    if agent.phase != Phase.IDLE:
        font = pygame.font.SysFont(None, 16)
        label = font.render(agent.action_type.name[:3], True, (255, 255, 255))
        surface.blit(label, (center[0] - label.get_width() // 2, center[1] - radius - 40))


class Renderer:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((_WINDOW_SIZE, _WINDOW_SIZE))
        pygame.display.set_caption("RLFighter")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)

    def draw(self, world: World, tick: int) -> None:
        self.screen.fill((30, 30, 30))

        # Arena border
        tl = _world_to_screen(np.array([0, ARENA_SIZE], dtype=np.float32))
        br = _world_to_screen(np.array([ARENA_SIZE, 0], dtype=np.float32))
        pygame.draw.rect(self.screen, (80, 80, 80), (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]), 2)

        # Hitboxes (draw behind agents)
        for agent in world.agents:
            _draw_hitbox(self.screen, agent)

        # Agents
        for agent in world.agents:
            _draw_agent(self.screen, agent)

        # HUD
        hud_lines = [
            f"Tick: {tick}",
        ]
        for tid, size in enumerate(world.team_sizes):
            alive = world.team_alive_count(tid)
            total_hp = sum(a.hp for a in world.agents if a.team_id == tid and a.alive)
            hud_lines.append(f"Team {tid}: {alive}/{size}  HP: {int(total_hp)}")

        y = 8
        for line in hud_lines:
            text = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(text, (8, y))
            y += 18

        # Controls hint
        hint = "WASD=move J=ver K=hor L=thrust Space=dodge H=heal"
        hint_surf = self.font.render(hint, True, (150, 150, 150))
        self.screen.blit(hint_surf, (8, _WINDOW_SIZE - 20))

        pygame.display.flip()

    def tick_fps(self, fps: int) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        pygame.quit()
