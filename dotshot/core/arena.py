"""Pure-NumPy arena geometry, circle-AABB collision, LOS raycasting, and 8-ray LIDAR."""

from typing import Iterable, Tuple
import numpy as np

from dotshot.core.config import RectObstacle


_LIDAR_ANGLES = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
_LIDAR_DIRS: Tuple[Tuple[float, float], ...] = tuple(
    (float(np.cos(a)), float(np.sin(a))) for a in _LIDAR_ANGLES
)


def clamp_to_arena(x: float, y: float, radius: float, width: float, height: float) -> Tuple[float, float]:
    cx = float(min(max(x, radius), width - radius))
    cy = float(min(max(y, radius), height - radius))
    return cx, cy


def resolve_circle_obstacles(
    x: float,
    y: float,
    radius: float,
    obstacles: Iterable[RectObstacle],
    width: float,
    height: float,
) -> Tuple[float, float, bool]:
    """Push a circle out of rectangular obstacles and arena boundaries."""
    hit_wall = False
    cx, cy = x, y

    for obs in obstacles:
        closest_x = min(max(cx, obs.x), obs.x + obs.w)
        closest_y = min(max(cy, obs.y), obs.y + obs.h)
        dx = cx - closest_x
        dy = cy - closest_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < radius * radius:
            hit_wall = True
            if dist_sq > 1e-8:
                dist = float(np.sqrt(dist_sq))
                overlap = radius - dist
                cx += (dx / dist) * overlap
                cy += (dy / dist) * overlap
            else:
                # Center inside rect: push out along nearest edge
                left_pen = abs(cx - obs.x)
                right_pen = abs((obs.x + obs.w) - cx)
                top_pen = abs(cy - obs.y)
                bot_pen = abs((obs.y + obs.h) - cy)
                min_pen = min(left_pen, right_pen, top_pen, bot_pen)
                if min_pen == left_pen:
                    cx = obs.x - radius
                elif min_pen == right_pen:
                    cx = obs.x + obs.w + radius
                elif min_pen == top_pen:
                    cy = obs.y - radius
                else:
                    cy = obs.y + obs.h + radius

    clamped_x, clamped_y = clamp_to_arena(cx, cy, radius, width, height)
    if clamped_x != cx or clamped_y != cy:
        hit_wall = True
    return clamped_x, clamped_y, hit_wall


def point_in_obstacles(x: float, y: float, radius: float, obstacles: Iterable[RectObstacle]) -> bool:
    for obs in obstacles:
        closest_x = min(max(x, obs.x), obs.x + obs.w)
        closest_y = min(max(y, obs.y), obs.y + obs.h)
        dx = x - closest_x
        dy = y - closest_y
        if dx * dx + dy * dy <= radius * radius:
            return True
    return False


def segment_intersects_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rect: RectObstacle,
) -> bool:
    """Slab method line-segment vs AABB intersection test."""
    dx = x2 - x1
    dy = y2 - y1
    t_min = 0.0
    t_max = 1.0

    for p, q_min, q_max in (
        (dx, rect.x - x1, rect.x + rect.w - x1),
        (dy, rect.y - y1, rect.y + rect.h - y1),
    ):
        if abs(p) < 1e-9:
            if q_min > 0.0 or q_max < 0.0:
                return False
        else:
            inv_p = 1.0 / p
            t1 = q_min * inv_p
            t2 = q_max * inv_p
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return True


def has_line_of_sight(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    obstacles: Iterable[RectObstacle],
) -> bool:
    """Return True if no LOS-blocking obstacle intersects segment (x1, y1)->(x2, y2)."""
    for obs in obstacles:
        if obs.blocks_los and segment_intersects_rect(x1, y1, x2, y2, obs):
            return False
    return True


def _ray_rect_distance(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    rect: RectObstacle,
    max_range: float,
) -> float:
    t_min = 0.0
    t_max = max_range

    for p, r_min, r_max in (
        (dx, rect.x - ox, rect.x + rect.w - ox),
        (dy, rect.y - oy, rect.y + rect.h - oy),
    ):
        if abs(p) < 1e-9:
            if r_min > 0.0 or r_max < 0.0:
                return max_range
        else:
            inv_p = 1.0 / p
            t1 = r_min * inv_p
            t2 = r_max * inv_p
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return max_range
    return t_min if t_min >= 0.0 else max_range


def compute_lidar_8(
    x: float,
    y: float,
    obstacles: Iterable[RectObstacle],
    width: float,
    height: float,
    max_range: float = 480.0,
) -> np.ndarray:
    """Compute 8 normalized obstacle/wall distances in [-1.0, 1.0]."""
    dists = np.empty(8, dtype=np.float32)
    inv_max = 1.0 / max_range
    for i, (dx, dy) in enumerate(_LIDAR_DIRS):
        best = max_range
        if dx > 1e-8:
            best = min(best, (width - x) / dx)
        elif dx < -1e-8:
            best = min(best, -x / dx)
        if dy > 1e-8:
            best = min(best, (height - y) / dy)
        elif dy < -1e-8:
            best = min(best, -y / dy)

        for obs in obstacles:
            best = min(best, _ray_rect_distance(x, y, dx, dy, obs, max_range))

        norm = max(0.0, min(1.0, best * inv_max))
        dists[i] = norm * 2.0 - 1.0
    return dists
