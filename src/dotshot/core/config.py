"""Configuration dataclasses and presets for DotShot Game v2."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SIM_VERSION = "2.2"

ELEMENT_WATER = 0
ELEMENT_GRASS = 1
ELEMENT_FIRE = 2

ELEMENT_NAMES: Dict[int, str] = {
    ELEMENT_WATER: "Water",
    ELEMENT_GRASS: "Grass",
    ELEMENT_FIRE: "Fire",
}

# Cyclic advantage: weapon_element -> effective_against_enemy_element
# Water beats Fire, Grass beats Water, Fire beats Grass
ELEMENT_COUNTERS: Dict[int, int] = {
    ELEMENT_WATER: ELEMENT_FIRE,
    ELEMENT_GRASS: ELEMENT_WATER,
    ELEMENT_FIRE: ELEMENT_GRASS,
}

# Reverse lookup: enemy_element -> counter_weapon_element
ELEMENT_WEAKNESS: Dict[int, int] = {v: k for k, v in ELEMENT_COUNTERS.items()}

ROLE_STRIKER = 0
ROLE_VANGUARD = 1
ROLE_SUPPORT = 2

ROLE_NAMES: Dict[int, str] = {
    ROLE_STRIKER: "Striker",
    ROLE_VANGUARD: "Vanguard",
    ROLE_SUPPORT: "Support",
}


@dataclass(frozen=True)
class RoleSpec:
    role_id: int
    name: str
    max_hp: float
    speed: float
    radius: float
    fire_cooldown: int
    ability_cooldown: int
    bullet_damage: float
    bullet_speed: float
    shattered_bonus_mult: float


ROLE_SPECS: Dict[int, RoleSpec] = {
    ROLE_STRIKER: RoleSpec(
        role_id=ROLE_STRIKER,
        name="Striker",
        max_hp=100.0,
        speed=4.8,
        radius=16.0,
        fire_cooldown=12,
        ability_cooldown=90,
        bullet_damage=34.0,
        bullet_speed=13.5,
        shattered_bonus_mult=1.35,
    ),
    ROLE_VANGUARD: RoleSpec(
        role_id=ROLE_VANGUARD,
        name="Vanguard",
        max_hp=180.0,
        speed=3.5,
        radius=20.0,
        fire_cooldown=16,
        ability_cooldown=140,
        bullet_damage=28.0,
        bullet_speed=11.5,
        shattered_bonus_mult=1.15,
    ),
    ROLE_SUPPORT: RoleSpec(
        role_id=ROLE_SUPPORT,
        name="Support",
        max_hp=115.0,
        speed=4.3,
        radius=16.0,
        fire_cooldown=14,
        ability_cooldown=120,
        bullet_damage=24.0,
        bullet_speed=12.5,
        shattered_bonus_mult=1.20,
    ),
}


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    w: float
    h: float
    blocks_los: bool = True
    kind: str = "pillar"


@dataclass
class StageConfig:
    stage: int
    env_id: str
    width: int = 1440
    height: int = 960
    max_steps: int = 1800
    num_agents: int = 1
    agent_roles: Tuple[int, ...] = (ROLE_STRIKER,)
    has_core: bool = False
    core_max_hp: float = 500.0
    allowed_enemy_types: Tuple[int, ...] = (ELEMENT_FIRE,)
    max_enemies: int = 5
    spawn_cooldown: int = 55
    enemy_speed_mult: float = 1.0
    enemy_can_shoot: bool = False
    spawn_allies: bool = False
    boss_enabled: bool = False
    boss_kill_interval: int = 8
    vision_range: float = 780.0
    partial_obs: bool = False
    reward_alpha: float = 0.5
    # "team": one shared team reward for every agent (CTDE default, ADR-005);
    # "mixed": per-agent (1 - alpha) * indiv + alpha * team (ablation only).
    reward_mode: str = "mixed"
    aim_bins: int = 32
    # v2.2 (ADR-006). Zero/neutral defaults keep Stages 1-2 single-life.
    respawn_ticks: int = 0  # >0: dead agents respawn near the core; all-dead no longer ends the episode
    win_bonus: float = 0.0  # raw team reward x Core HP ratio for reaching max_steps with the Core alive
    victory_terminal: bool = False  # reaching max_steps is a terminal victory, not a time-limit truncation
    enemy_bullet_damage: float = 10.0  # bosses deal 1.4x
    agent_contact_mult: float = 0.04  # contact damage per tick = enemy.contact_damage_rate * mult
    core_contact_mult: float = 0.045
    max_bullets_per_target: int = 2  # own bullets in flight toward one enemy
    fire_element_credit: float = 1.5  # raw reward at fire time: +c for the counter element, -c/2 for a wrong one
    action_repeat: int = 8  # ticks per decision; the action is held and rewards summed (ADR-007)
    guard_potential_coef: float = 0.0  # raw; >0 adds potential-based shaping toward the Core (team mode)
    shaping_gamma: float = 0.99  # must equal the learner's per-decision discount
    aim_shaping_anneal_steps: int = 0  # >0: aim shaping decays linearly to 0 over this many env steps
    obstacles: Tuple[RectObstacle, ...] = field(default_factory=tuple)


SINGLE_STAGE_OBSTACLES: Tuple[RectObstacle, ...] = (
    RectObstacle(360.0, 220.0, 110.0, 110.0),
    RectObstacle(970.0, 220.0, 110.0, 110.0),
    RectObstacle(360.0, 630.0, 110.0, 110.0),
    RectObstacle(970.0, 630.0, 110.0, 110.0),
)

COOP_STAGE_OBSTACLES: Tuple[RectObstacle, ...] = (
    RectObstacle(430.0, 210.0, 120.0, 80.0),
    RectObstacle(890.0, 210.0, 120.0, 80.0),
    RectObstacle(430.0, 670.0, 120.0, 80.0),
    RectObstacle(890.0, 670.0, 120.0, 80.0),
    RectObstacle(250.0, 420.0, 80.0, 120.0),
    RectObstacle(1110.0, 420.0, 80.0, 120.0),
)


STAGE_CONFIGS: Dict[int, StageConfig] = {
    1: StageConfig(
        stage=1,
        env_id="DotShot-Stage1-v1",
        max_steps=1500,
        num_agents=1,
        agent_roles=(ROLE_STRIKER,),
        has_core=False,
        allowed_enemy_types=(ELEMENT_FIRE,),
        max_enemies=4,
        spawn_cooldown=48,
        enemy_speed_mult=0.95,
        enemy_can_shoot=False,
        spawn_allies=False,
        boss_enabled=False,
        vision_range=1600.0,
        partial_obs=False,
        reward_alpha=0.0,
        obstacles=SINGLE_STAGE_OBSTACLES,
    ),
    2: StageConfig(
        stage=2,
        env_id="DotShot-Stage2-v1",
        max_steps=1800,
        num_agents=1,
        agent_roles=(ROLE_STRIKER,),
        has_core=False,
        allowed_enemy_types=(ELEMENT_WATER, ELEMENT_GRASS, ELEMENT_FIRE),
        max_enemies=6,
        spawn_cooldown=42,
        enemy_speed_mult=1.05,
        enemy_can_shoot=True,
        spawn_allies=True,
        boss_enabled=False,
        vision_range=1600.0,
        partial_obs=False,
        reward_alpha=0.0,
        obstacles=SINGLE_STAGE_OBSTACLES,
    ),
    3: StageConfig(
        stage=3,
        env_id="DotShot-Stage3-v1",
        max_steps=2000,
        num_agents=2,
        agent_roles=(ROLE_STRIKER, ROLE_STRIKER),
        has_core=True,
        core_max_hp=500.0,
        allowed_enemy_types=(ELEMENT_WATER, ELEMENT_GRASS, ELEMENT_FIRE),
        max_enemies=8,
        spawn_cooldown=36,
        enemy_speed_mult=1.10,
        enemy_can_shoot=True,
        spawn_allies=False,
        boss_enabled=True,
        boss_kill_interval=8,
        vision_range=760.0,
        partial_obs=True,
        reward_alpha=0.65,
        reward_mode="team",
        respawn_ticks=150,
        win_bonus=20.0,
        victory_terminal=True,
        obstacles=COOP_STAGE_OBSTACLES,
    ),
    4: StageConfig(
        stage=4,
        env_id="DotShot-Stage4-v1",
        max_steps=2200,
        num_agents=3,
        agent_roles=(ROLE_VANGUARD, ROLE_STRIKER, ROLE_SUPPORT),
        has_core=True,
        core_max_hp=550.0,
        allowed_enemy_types=(ELEMENT_WATER, ELEMENT_GRASS, ELEMENT_FIRE),
        max_enemies=10,
        spawn_cooldown=30,
        enemy_speed_mult=1.18,
        enemy_can_shoot=True,
        spawn_allies=False,
        boss_enabled=True,
        boss_kill_interval=7,
        vision_range=760.0,
        partial_obs=True,
        reward_alpha=0.70,
        reward_mode="team",
        respawn_ticks=150,
        win_bonus=20.0,
        victory_terminal=True,
        enemy_bullet_damage=17.0,  # calibrated: reference script wins 80.2% of 96 val episodes
        obstacles=COOP_STAGE_OBSTACLES,
    ),
}


def get_stage_config(stage: int) -> StageConfig:
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unsupported DotShot stage: {stage}. Expected one of {list(STAGE_CONFIGS)}.")
    return STAGE_CONFIGS[stage]
