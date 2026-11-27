"""Simulation entities for DotShot Game v2 (pure NumPy/Python, no Pygame dependency)."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from dotshot.core.config import (
    ELEMENT_FIRE,
    ELEMENT_GRASS,
    ELEMENT_NAMES,
    ELEMENT_WATER,
    ROLE_SPECS,
    ROLE_STRIKER,
)


@dataclass
class CentralCore:
    x: float
    y: float
    max_hp: float = 500.0
    hp: float = 500.0
    radius: float = 38.0
    under_attack_timer: int = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0.0

    def take_damage(self, amount: float) -> float:
        prev = self.hp
        self.hp = max(0.0, self.hp - amount)
        self.under_attack_timer = 20
        return prev - self.hp

    def heal(self, amount: float) -> float:
        if self.hp <= 0.0:
            return 0.0
        prev = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - prev


class AgentState:
    """State of a controllable DotShot agent."""

    def __init__(self, idx: int, x: float, y: float, role_id: int = ROLE_STRIKER):
        spec = ROLE_SPECS[role_id]
        self.idx = idx
        self.role_id = role_id
        self.role_name = spec.name
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.radius = spec.radius
        self.speed = spec.speed
        self.max_hp = spec.max_hp
        self.hp = spec.max_hp
        self.fire_cooldown_max = spec.fire_cooldown
        self.ability_cooldown_max = spec.ability_cooldown
        self.bullet_damage = spec.bullet_damage
        self.bullet_speed = spec.bullet_speed
        self.shattered_bonus_mult = spec.shattered_bonus_mult

        self.fire_cooldown = 0
        self.ability_cooldown = 0
        self.shield_active_timer = 0
        self.repair_pulse_timer = 0
        self.alive = True
        self.last_weapon = -1
        self.aim_dx = 1.0
        self.aim_dy = 0.0
        self.spawn_x, self.spawn_y = self.x, self.y
        self.respawn_timer = 0  # ticks until a dead agent respawns (v2.2 core stages)

    def respawn(self) -> None:
        """Return to the spawn point with full HP and cleared timers."""
        self.x, self.y = self.spawn_x, self.spawn_y
        self.vx = self.vy = 0.0
        self.hp = self.max_hp
        self.fire_cooldown = self.ability_cooldown = 0
        self.shield_active_timer = self.repair_pulse_timer = 0
        self.respawn_timer = 0
        self.alive = True

    def take_damage(self, amount: float) -> float:
        if not self.alive:
            return 0.0
        if self.shield_active_timer > 0:
            amount *= 0.35
        prev = self.hp
        self.hp = max(0.0, self.hp - amount)
        if self.hp <= 0.0:
            self.alive = False
            self.vx = 0.0
            self.vy = 0.0
        return prev - self.hp

    def heal(self, amount: float) -> float:
        if not self.alive:
            return 0.0
        prev = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - prev


class Projectile:
    """Agent-fired elemental projectile."""

    def __init__(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        element_type: int,
        owner_idx: int,
        damage: float,
        speed: float,
        target_uid: Optional[int] = None,
        well_aimed: bool = False,
    ):
        self.x = float(x)
        self.y = float(y)
        self.dx = float(dx)
        self.dy = float(dy)
        self.element_type = element_type
        self.owner_idx = owner_idx
        self.damage = float(damage)
        self.speed = float(speed)
        self.radius = 5.5
        self.active = True
        self.target_uid = target_uid
        self.well_aimed = well_aimed

    def step(self) -> None:
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class EnemyProjectile:
    """Dodgeable hostile projectile."""

    def __init__(self, x: float, y: float, dx: float, dy: float, element_type: int, damage: float = 12.0):
        self.x = float(x)
        self.y = float(y)
        self.dx = float(dx)
        self.dy = float(dy)
        self.element_type = element_type
        self.damage = float(damage)
        self.speed = 5.4
        self.radius = 5.5
        self.active = True

    def step(self) -> None:
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class Enemy:
    """Hostile elemental unit or Shielded Boss."""

    def __init__(
        self,
        uid: int,
        x: float,
        y: float,
        enemy_type: int,
        speed_mult: float = 1.0,
        difficulty: float = 1.0,
        is_boss: bool = False,
        shoot_cooldown_init: int = 90,
    ):
        self.uid = uid
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.enemy_type = enemy_type
        self.element_name = ELEMENT_NAMES[enemy_type]
        self.is_boss = is_boss

        if enemy_type == ELEMENT_WATER:
            base_hp, base_speed, base_dmg = 95.0, 2.0, 18.0
        elif enemy_type == ELEMENT_GRASS:
            base_hp, base_speed, base_dmg = 140.0, 1.35, 24.0
        else:  # ELEMENT_FIRE
            base_hp, base_speed, base_dmg = 72.0, 2.75, 15.0

        diff_hp_scale = 1.0 + (difficulty - 1.0) * 0.12
        diff_spd_scale = 1.0 + (difficulty - 1.0) * 0.06

        if is_boss:
            self.radius = 28.0
            self.max_hp = base_hp * 2.8 * diff_hp_scale
            self.max_shield = 110.0 * diff_hp_scale
            self.shield_hp = self.max_shield
            self.speed = base_speed * 0.72 * speed_mult * diff_spd_scale
            self.contact_damage_rate = base_dmg * 1.4
        else:
            self.radius = 16.0
            self.max_hp = base_hp * diff_hp_scale
            self.max_shield = 0.0
            self.shield_hp = 0.0
            self.speed = base_speed * speed_mult * diff_spd_scale
            self.contact_damage_rate = base_dmg

        self.hp = self.max_hp
        self.alive = True
        self.marked_timer = 0
        self.shield_broken_timer = 0
        self.shoot_cooldown = shoot_cooldown_init

    def take_hull_damage(self, amount: float) -> float:
        if not self.alive:
            return 0.0
        mult = 1.0
        if self.marked_timer > 0:
            mult *= 1.35
        if self.is_boss and self.shield_hp > 0.0:
            # Unbroken elemental shield mitigates hull damage heavily
            mult *= 0.10
        elif self.is_boss and self.shield_broken_timer > 0:
            mult *= 1.35
        actual = amount * mult
        prev = self.hp
        self.hp = max(0.0, self.hp - actual)
        if self.hp <= 0.0:
            self.alive = False
        return prev - self.hp

    def take_shield_damage(self, amount: float) -> bool:
        """Apply counter-element damage to boss shield. Returns True on shield shatter."""
        if not self.is_boss or self.shield_hp <= 0.0:
            return False
        self.shield_hp = max(0.0, self.shield_hp - amount)
        if self.shield_hp <= 0.0:
            self.shield_broken_timer = 160
            return True
        return False


class NeutralAlly:
    """Non-hostile NPC unit traversing the arena (friendly-fire hazard in Stage 2)."""

    def __init__(self, x: float, y: float, tx: float, ty: float, rng: np.random.Generator):
        self.x = float(x)
        self.y = float(y)
        self.tx = float(tx)
        self.ty = float(ty)
        self.spawn_x = float(x)
        self.spawn_y = float(y)
        self.radius = 15.0
        self.max_hp = 50.0
        self.hp = 50.0
        self.speed = 1.7
        self.alive = True
        self.angle = float(np.arctan2(ty - y, tx - x))
        self.rng = rng

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)
        if self.hp <= 0.0:
            self.alive = False

    def step(self, width: float, height: float) -> None:
        target_angle = float(np.arctan2(self.ty - self.y, self.tx - self.x))
        diff = (target_angle - self.angle + np.pi) % (2.0 * np.pi) - np.pi
        self.angle += diff * 0.06 + float(self.rng.uniform(-0.12, 0.12))
        self.x += float(np.cos(self.angle) * self.speed)
        self.y += float(np.sin(self.angle) * self.speed)

        if self.x < -80.0 or self.x > width + 80.0 or self.y < -80.0 or self.y > height + 80.0:
            self.alive = False
        elif (self.x - self.tx) ** 2 + (self.y - self.ty) ** 2 < 36.0 ** 2:
            self.alive = False
