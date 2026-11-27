"""High-FPS Pygame tactical renderer for DotShot Game v2 with MARL explainability overlay."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pygame

from dotshot.core.arena import compute_lidar_8
from dotshot.core.config import (
    ELEMENT_FIRE,
    ELEMENT_GRASS,
    ELEMENT_WATER,
    ROLE_SUPPORT,
    ROLE_VANGUARD,
)


# Okabe-Ito / tactical neon palette
BG_COLOR = (11, 15, 23)
GRID_COLOR = (20, 28, 43)
COVER_FILL = (30, 41, 59)
COVER_BORDER = (71, 85, 105)
COVER_SHADOW = (6, 9, 15)

ELEMENT_COLORS: Dict[int, Tuple[int, int, int]] = {
    ELEMENT_WATER: (56, 189, 248),
    ELEMENT_GRASS: (34, 197, 94),
    ELEMENT_FIRE: (249, 115, 22),
}

ROLE_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (226, 232, 240),  # Striker: crisp silver-white
    1: (56, 189, 248),   # Vanguard: cyan-steel
    2: (74, 222, 128),   # Support: emerald-mint
}


class DotShotPygameRenderer:
    """Decoupled Pygame renderer with pre-baked glow sprites and Tab MARL overlay."""

    def __init__(self, width: int = 1440, height: int = 960, render_mode: str = "human"):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.render_mode = render_mode

        if render_mode == "human":
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.SCALED)
            pygame.display.set_caption("DotShot Game v2 - Tactical MARL Simulator")
        else:
            self.screen = pygame.Surface((width, height))

        self.canvas = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()

        try:
            self.font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
            self.font_hud = pygame.font.SysFont("Segoe UI", 15, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 13, bold=True)
            self.font_callout = pygame.font.SysFont("Segoe UI", 16, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 28)
            self.font_hud = pygame.font.Font(None, 20)
            self.font_small = pygame.font.Font(None, 16)
            self.font_callout = pygame.font.Font(None, 22)

        self._glow_cache: Dict[Tuple[Tuple[int, int, int], int, int, int], pygame.Surface] = {}
        self._static_bg_cache: Optional[pygame.Surface] = None
        self._cached_obstacles_key: Optional[Tuple[Any, ...]] = None

        self.particles: List[Dict[str, Any]] = []
        self.callouts: List[Dict[str, Any]] = []
        self.shake_timer = 0
        self.shake_intensity = 0
        self.show_marl_overlay = True
        self.quit_requested = False
        # Own RNG: drawing must never consume the simulator's random stream.
        self.rng = np.random.default_rng(0)

    def _get_glow_sprite(self, color: Tuple[int, int, int], radius: int, glow_radius: int, alpha: int) -> pygame.Surface:
        key = (color, radius, glow_radius, alpha)
        cached = self._glow_cache.get(key)
        if cached is not None:
            return cached

        size = (radius + glow_radius) * 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx = radius + glow_radius
        cy = radius + glow_radius
        for r in range(glow_radius, 0, -2):
            a = int(alpha * (1.0 - r / max(1, glow_radius)))
            pygame.draw.circle(surf, (*color, a), (cx, cy), radius + r)
        pygame.draw.circle(surf, (*color, 255), (cx, cy), radius)
        self._glow_cache[key] = surf
        return surf

    def _draw_glow(self, target: pygame.Surface, color: Tuple[int, int, int], pos: Tuple[float, float], radius: int, glow_radius: int = 12, alpha: int = 75) -> None:
        sprite = self._get_glow_sprite(color, radius, glow_radius, alpha)
        half = radius + glow_radius
        target.blit(sprite, (int(pos[0]) - half, int(pos[1]) - half))

    def _ensure_static_background(self, engine: Any) -> pygame.Surface:
        obs_key = tuple((o.x, o.y, o.w, o.h) for o in engine.cfg.obstacles)
        if self._static_bg_cache is not None and self._cached_obstacles_key == obs_key:
            return self._static_bg_cache

        bg = pygame.Surface((self.width, self.height))
        bg.fill(BG_COLOR)

        # Tactical grid
        grid = 48
        for x in range(0, self.width, grid):
            pygame.draw.line(bg, GRID_COLOR, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid):
            pygame.draw.line(bg, GRID_COLOR, (0, y), (self.width, y), 1)

        # Arena perimeter accent
        pygame.draw.rect(bg, (30, 41, 59), pygame.Rect(2, 2, self.width - 4, self.height - 4), 2)

        # Beveled cover obstacles with drop shadow
        for ob in engine.cfg.obstacles:
            shadow_rect = pygame.Rect(int(ob.x + 6), int(ob.y + 6), int(ob.w), int(ob.h))
            pygame.draw.rect(bg, COVER_SHADOW, shadow_rect, border_radius=6)
            rect = pygame.Rect(int(ob.x), int(ob.y), int(ob.w), int(ob.h))
            pygame.draw.rect(bg, COVER_FILL, rect, border_radius=6)
            pygame.draw.rect(bg, COVER_BORDER, rect, 2, border_radius=6)
            inner = rect.inflate(-10, -10)
            pygame.draw.rect(bg, (38, 52, 74), inner, 1, border_radius=4)

        self._static_bg_cache = bg
        self._cached_obstacles_key = obs_key
        return bg

    def _ingest_events(self, engine: Any) -> None:
        rng = self.rng
        for ev in getattr(engine, "visual_events", []):
            etype = ev.get("type")
            if etype == "spark" and len(self.particles) < 380:
                col = ELEMENT_COLORS.get(ev.get("element", 0), (226, 232, 240))
                for _ in range(ev.get("count", 6)):
                    ang = float(rng.uniform(0.0, 2.0 * np.pi))
                    spd = float(rng.uniform(1.5, 4.8))
                    self.particles.append({
                        "x": ev["x"],
                        "y": ev["y"],
                        "vx": np.cos(ang) * spd,
                        "vy": np.sin(ang) * spd,
                        "color": col,
                        "life": 18,
                        "max_life": 18,
                        "size": 3.0,
                    })
            elif etype == "callout":
                self.callouts.append({
                    "x": ev["x"],
                    "y": ev["y"],
                    "text": ev["text"],
                    "color": ev.get("color", (248, 250, 252)),
                    "life": 42,
                    "max_life": 42,
                })
            elif etype == "shake":
                self.shake_timer = 10
                self.shake_intensity = max(self.shake_intensity, int(ev.get("intensity", 5)))

    def _draw_polygon_shape(self, surf: pygame.Surface, color: Tuple[int, int, int], center: Tuple[float, float], radius: float, sides: int, angle_offset: float = 0.0, width: int = 0) -> None:
        pts = []
        for k in range(sides):
            ang = angle_offset + (2.0 * np.pi * k) / sides
            pts.append((center[0] + np.cos(ang) * radius, center[1] + np.sin(ang) * radius))
        pygame.draw.polygon(surf, color, pts, width)

    def render(self, engine: Any, overlay_info: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    self.show_marl_overlay = not self.show_marl_overlay

        self._ingest_events(engine)
        bg = self._ensure_static_background(engine)
        self.canvas.blit(bg, (0, 0))

        # 1. Central Energy Core
        if engine.core is not None:
            core = engine.core
            pulse = 4 + int(3 * np.sin(engine.step_count * 0.12))
            core_col = (248, 113, 113) if core.under_attack_timer > 0 else (56, 189, 248)
            self._draw_glow(self.canvas, core_col, (core.x, core.y), int(core.radius), glow_radius=16 + pulse, alpha=70)
            pygame.draw.circle(self.canvas, (15, 23, 42), (int(core.x), int(core.y)), int(core.radius - 6))
            pygame.draw.circle(self.canvas, core_col, (int(core.x), int(core.y)), int(core.radius), 3)
            hp_ratio = max(0.0, core.hp / max(1.0, core.max_hp))
            arc_rect = pygame.Rect(int(core.x - core.radius - 8), int(core.y - core.radius - 8), int((core.radius + 8) * 2), int((core.radius + 8) * 2))
            pygame.draw.arc(self.canvas, (34, 197, 94), arc_rect, np.pi * 0.5, np.pi * 0.5 + 2.0 * np.pi * hp_ratio, 4)

        # 2. Particles
        kept_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] > 0:
                ratio = p["life"] / p["max_life"]
                sz = max(1, int(p["size"] * ratio))
                pygame.draw.circle(self.canvas, p["color"], (int(p["x"]), int(p["y"])), sz)
                kept_particles.append(p)
        self.particles = kept_particles

        # 3. Neutral Allies (Stage 2)
        for ally in engine.neutral_allies:
            if not ally.alive:
                continue
            self._draw_glow(self.canvas, (226, 232, 240), (ally.x, ally.y), int(ally.radius), glow_radius=10, alpha=50)
            pygame.draw.circle(self.canvas, (248, 250, 252), (int(ally.x), int(ally.y)), int(ally.radius), 2)

        # 4. Bullets
        for b in engine.bullets:
            col = ELEMENT_COLORS[b.element_type]
            self._draw_glow(self.canvas, col, (b.x, b.y), int(b.radius), glow_radius=8, alpha=110)
            tail = (b.x - b.dx * 12, b.y - b.dy * 12)
            pygame.draw.line(self.canvas, col, (int(b.x), int(b.y)), (int(tail[0]), int(tail[1])), 3)

        for eb in engine.enemy_bullets:
            col = (251, 146, 60)
            self._draw_glow(self.canvas, col, (eb.x, eb.y), int(eb.radius), glow_radius=7, alpha=95)
            pygame.draw.circle(self.canvas, (254, 215, 170), (int(eb.x), int(eb.y)), 3)

        # 5. Enemies & Shielded Bosses
        for enemy in engine.enemies:
            if not enemy.alive:
                continue
            col = ELEMENT_COLORS[enemy.enemy_type]
            self._draw_glow(self.canvas, col, (enemy.x, enemy.y), int(enemy.radius), glow_radius=14 if enemy.is_boss else 9, alpha=70)

            # Distinct geometry per element for colorblind clarity
            if enemy.enemy_type == ELEMENT_WATER:
                pygame.draw.circle(self.canvas, col, (int(enemy.x), int(enemy.y)), int(enemy.radius))
                pygame.draw.circle(self.canvas, (240, 249, 255), (int(enemy.x), int(enemy.y)), int(enemy.radius), 2)
            elif enemy.enemy_type == ELEMENT_GRASS:
                self._draw_polygon_shape(self.canvas, col, (enemy.x, enemy.y), enemy.radius + 2, 4, angle_offset=np.pi * 0.25)
                self._draw_polygon_shape(self.canvas, (240, 253, 244), (enemy.x, enemy.y), enemy.radius + 2, 4, angle_offset=np.pi * 0.25, width=2)
            else:
                self._draw_polygon_shape(self.canvas, col, (enemy.x, enemy.y), enemy.radius + 2, 3, angle_offset=-np.pi * 0.5)
                self._draw_polygon_shape(self.canvas, (255, 247, 237), (enemy.x, enemy.y), enemy.radius + 2, 3, angle_offset=-np.pi * 0.5, width=2)

            if enemy.marked_timer > 0:
                pygame.draw.circle(self.canvas, (250, 204, 21), (int(enemy.x), int(enemy.y)), int(enemy.radius + 6), 2)

            # HP and Boss Shield Bars
            bw = int(enemy.radius * 2.4)
            bx = int(enemy.x - bw * 0.5)
            by = int(enemy.y - enemy.radius - 12)
            pygame.draw.rect(self.canvas, (30, 41, 59), (bx, by, bw, 5))
            hp_w = int(bw * max(0.0, enemy.hp / max(1.0, enemy.max_hp)))
            pygame.draw.rect(self.canvas, (239, 68, 68), (bx, by, hp_w, 5))

            if enemy.is_boss and enemy.max_shield > 0.0:
                sy = by - 7
                pygame.draw.rect(self.canvas, (30, 41, 59), (bx, sy, bw, 4))
                sh_w = int(bw * max(0.0, enemy.shield_hp / enemy.max_shield))
                pygame.draw.rect(self.canvas, col, (bx, sy, sh_w, 4))

        # 6. Agents & Explainability Overlay
        if self.show_marl_overlay and overlay_info and "attention_links" in overlay_info:
            for i, j, weight in overlay_info["attention_links"]:
                if i < len(engine.agents) and j < len(engine.agents):
                    a1, a2 = engine.agents[i], engine.agents[j]
                    if a1.alive and a2.alive:
                        pygame.draw.line(self.canvas, (99, 102, 241), (int(a1.x), int(a1.y)), (int(a2.x), int(a2.y)), max(1, int(weight * 4)))

        for agent in engine.agents:
            if not agent.alive:
                continue

            if self.show_marl_overlay:
                # 8-ray LIDAR visualization
                lidar = compute_lidar_8(agent.x, agent.y, engine.cfg.obstacles, self.width, self.height, max_range=480.0)
                for r_i, val in enumerate(lidar):
                    ang = r_i * (2.0 * np.pi / 8.0)
                    dist = ((float(val) + 1.0) * 0.5) * 480.0
                    ex = agent.x + np.cos(ang) * dist
                    ey = agent.y + np.sin(ang) * dist
                    pygame.draw.line(self.canvas, (30, 58, 95), (int(agent.x), int(agent.y)), (int(ex), int(ey)), 1)

                # Aim laser line
                lx = agent.x + agent.aim_dx * 220.0
                ly = agent.y + agent.aim_dy * 220.0
                aim_col = ELEMENT_COLORS.get(agent.last_weapon, (148, 163, 184))
                pygame.draw.line(self.canvas, aim_col, (int(agent.x), int(agent.y)), (int(lx), int(ly)), 1)

            base_col = ROLE_COLORS.get(agent.role_id, (226, 232, 240))
            weapon_ring = ELEMENT_COLORS.get(agent.last_weapon, base_col)
            self._draw_glow(self.canvas, base_col, (agent.x, agent.y), int(agent.radius), glow_radius=12, alpha=65)

            if agent.shield_active_timer > 0:
                pygame.draw.circle(self.canvas, (56, 189, 248), (int(agent.x), int(agent.y)), int(agent.radius + 10), 2)
            if agent.repair_pulse_timer > 0:
                pulse_r = int(agent.radius + (22 - agent.repair_pulse_timer) * 14)
                pygame.draw.circle(self.canvas, (34, 197, 94), (int(agent.x), int(agent.y)), pulse_r, 2)

            aim_ang = float(np.arctan2(agent.aim_dy, agent.aim_dx))
            if agent.role_id == ROLE_VANGUARD:
                self._draw_polygon_shape(self.canvas, base_col, (agent.x, agent.y), agent.radius, 6, angle_offset=aim_ang)
            elif agent.role_id == ROLE_SUPPORT:
                self._draw_polygon_shape(self.canvas, base_col, (agent.x, agent.y), agent.radius, 4, angle_offset=aim_ang)
            else:
                # Striker directional chevron
                tip = (agent.x + agent.aim_dx * (agent.radius + 5), agent.y + agent.aim_dy * (agent.radius + 5))
                left_ang = aim_ang + 2.35
                right_ang = aim_ang - 2.35
                p2 = (agent.x + np.cos(left_ang) * agent.radius, agent.y + np.sin(left_ang) * agent.radius)
                p3 = (agent.x + np.cos(right_ang) * agent.radius, agent.y + np.sin(right_ang) * agent.radius)
                pygame.draw.polygon(self.canvas, base_col, [tip, p2, (agent.x, agent.y), p3])

            pygame.draw.circle(self.canvas, weapon_ring, (int(agent.x), int(agent.y)), int(agent.radius + 3), 2)

            if self.show_marl_overlay and overlay_info and "order" in overlay_info:
                order = overlay_info["order"]
                if agent.idx in order:
                    rank = order.index(agent.idx) + 1
                    badge = self.font_small.render(f"#{rank}", True, (250, 204, 21))
                    self.canvas.blit(badge, (int(agent.x - 10), int(agent.y - agent.radius - 22)))

        # 7. Floating Callouts
        kept_callouts = []
        for c in self.callouts:
            c["y"] -= 0.7
            c["life"] -= 1
            if c["life"] > 0:
                txt = self.font_callout.render(c["text"], True, c["color"])
                self.canvas.blit(txt, (int(c["x"] - txt.get_width() * 0.5), int(c["y"])))
                kept_callouts.append(c)
        self.callouts = kept_callouts

        # 8. Translucent HUD
        self._draw_hud(engine)

        # Apply damped screen shake when blitting to screen
        ox, oy = 0, 0
        if self.shake_timer > 0:
            self.shake_timer -= 1
            ox = int(self.rng.integers(-self.shake_intensity, self.shake_intensity + 1))
            oy = int(self.rng.integers(-self.shake_intensity, self.shake_intensity + 1))
            if self.shake_timer == 0:
                self.shake_intensity = 0

        self.screen.fill(BG_COLOR)
        self.screen.blit(self.canvas, (ox, oy))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
            return None

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).copy()

    def _draw_hud(self, engine: Any) -> None:
        bar = pygame.Surface((self.width, 44), pygame.SRCALPHA)
        bar.fill((15, 23, 42, 200))
        self.canvas.blit(bar, (0, 0))

        core_str = f"CORE: {int(engine.core.hp)}/{int(engine.core.max_hp)}" if engine.core is not None else "MODE: SOLO"
        hud_text = (
            f"STAGE {engine.cfg.stage}   |   WAVE {engine.wave_index}   |   "
            f"SCORE {int(engine.telemetry['score'])}   |   KILLS {int(engine.telemetry['kills'])}   |   "
            f"{core_str}   |   STEP {engine.step_count}/{engine.cfg.max_steps}"
        )
        surf = self.font_hud.render(hud_text, True, (241, 245, 249))
        self.canvas.blit(surf, (20, 12))

        fps_txt = self.font_small.render(f"FPS {int(self.clock.get_fps())} | TAB: Overlay {'ON' if self.show_marl_overlay else 'OFF'}", True, (148, 163, 184))
        self.canvas.blit(fps_txt, (self.width - fps_txt.get_width() - 20, 14))

        # Bottom agent cards
        card_w, card_h = 210, 48
        gap = 14
        total_w = len(engine.agents) * card_w + max(0, len(engine.agents) - 1) * gap
        start_x = (self.width - total_w) // 2
        y = self.height - card_h - 10
        for i, ag in enumerate(engine.agents):
            x = start_x + i * (card_w + gap)
            card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            card.fill((15, 23, 42, 205))
            self.canvas.blit(card, (x, y))
            border = ROLE_COLORS.get(ag.role_id, (148, 163, 184)) if ag.alive else (100, 116, 139)
            pygame.draw.rect(self.canvas, border, (x, y, card_w, card_h), 1, border_radius=6)

            label = self.font_small.render(f"A{ag.idx} {ag.role_name} {'[DEAD]' if not ag.alive else ''}", True, (241, 245, 249))
            self.canvas.blit(label, (x + 10, y + 6))

            hp_ratio = max(0.0, ag.hp / max(1.0, ag.max_hp))
            pygame.draw.rect(self.canvas, (30, 41, 59), (x + 10, y + 28, card_w - 20, 8), border_radius=3)
            pygame.draw.rect(self.canvas, (34, 197, 94) if hp_ratio > 0.35 else (239, 68, 68), (x + 10, y + 28, int((card_w - 20) * hp_ratio), 8), border_radius=3)

    def close(self) -> None:
        pygame.quit()
