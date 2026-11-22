"""Capture deterministic 1440x960 in-game screenshots across all 4 DotShot Game v2 stages."""

import os
import pathlib
import sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from dotshot.core.config import (
    ELEMENT_FIRE,
    ELEMENT_GRASS,
    ELEMENT_WATER,
)
from dotshot.core.entities import Enemy, EnemyProjectile, NeutralAlly, Projectile
from dotshot.envs.multi_agent import DotShotParallelEnv
from dotshot.envs.single_agent import DotShotSingleEnv


OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets" / "screenshots"


def _save_frame(frame: np.ndarray, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    Image.fromarray(frame).save(path)
    print(f"Saved {path} ({path.stat().st_size // 1024} KB)")


def capture_stage1() -> None:
    env = DotShotSingleEnv(stage=1, render_mode="rgb_array")
    env.reset(seed=10)
    eng = env.engine
    eng.enemies = [
        Enemy(uid=101, x=480.0, y=380.0, enemy_type=ELEMENT_FIRE),
        Enemy(uid=102, x=920.0, y=540.0, enemy_type=ELEMENT_FIRE),
        Enemy(uid=103, x=640.0, y=220.0, enemy_type=ELEMENT_FIRE),
    ]
    ag = eng.agents[0]
    ag.x, ag.y = 660.0, 500.0
    ag.aim_dx, ag.aim_dy = -0.78, -0.62
    ag.last_weapon = ELEMENT_WATER
    eng.bullets = [
        Projectile(ag.x - 45.0, ag.y - 36.0, -0.78, -0.62, ELEMENT_WATER, 0, 34.0, 13.5),
        Projectile(ag.x - 110.0, ag.y - 88.0, -0.78, -0.62, ELEMENT_WATER, 0, 34.0, 13.5),
    ]
    eng.visual_events.append({"type": "spark", "x": 500.0, "y": 395.0, "element": ELEMENT_WATER, "count": 12})
    eng.telemetry["score"] = 400.0
    eng.telemetry["kills"] = 4.0
    eng.step_count = 340
    frame = env.render()
    _save_frame(frame, "stage1_cover_combat.png")
    env.close()


def capture_stage2() -> None:
    env = DotShotSingleEnv(stage=2, render_mode="rgb_array")
    env.reset(seed=20)
    eng = env.engine
    ag = eng.agents[0]
    ag.x, ag.y = 700.0, 510.0
    ag.aim_dx, ag.aim_dy = 0.92, -0.39
    ag.last_weapon = ELEMENT_GRASS
    eng.enemies = [
        Enemy(uid=201, x=980.0, y=390.0, enemy_type=ELEMENT_WATER),
        Enemy(uid=202, x=460.0, y=610.0, enemy_type=ELEMENT_GRASS),
        Enemy(uid=203, x=830.0, y=260.0, enemy_type=ELEMENT_FIRE),
    ]
    eng.neutral_allies = [NeutralAlly(540.0, 360.0, 950.0, 620.0, eng.np_random)]
    eng.bullets = [Projectile(ag.x + 80.0, ag.y - 34.0, 0.92, -0.39, ELEMENT_GRASS, 0, 34.0, 13.5)]
    eng.enemy_bullets = [
        EnemyProjectile(880.0, 440.0, -0.85, 0.52, ELEMENT_WATER),
        EnemyProjectile(540.0, 580.0, 0.85, -0.52, ELEMENT_GRASS),
    ]
    eng.telemetry["score"] = 950.0
    eng.telemetry["kills"] = 9.0
    eng.step_count = 620
    frame = env.render()
    _save_frame(frame, "stage2_elemental_mastery.png")
    env.close()


def capture_stage3() -> None:
    env = DotShotParallelEnv(stage=3, render_mode="rgb_array")
    env.reset(seed=30)
    eng = env.engine
    eng.core.hp = 415.0
    eng.agents[0].x, eng.agents[0].y = 590.0, 440.0
    eng.agents[0].aim_dx, eng.agents[0].aim_dy = -0.86, -0.51
    eng.agents[0].last_weapon = ELEMENT_GRASS

    eng.agents[1].x, eng.agents[1].y = 850.0, 540.0
    eng.agents[1].aim_dx, eng.agents[1].aim_dy = 0.88, -0.47
    eng.agents[1].last_weapon = ELEMENT_WATER

    boss = Enemy(uid=301, x=350.0, y=300.0, enemy_type=ELEMENT_WATER, is_boss=True)
    boss.shield_hp = 55.0
    boss.hp = 210.0
    eng.enemies = [
        boss,
        Enemy(uid=302, x=1060.0, y=420.0, enemy_type=ELEMENT_FIRE),
        Enemy(uid=303, x=960.0, y=720.0, enemy_type=ELEMENT_GRASS),
    ]
    eng.bullets = [
        Projectile(480.0, 375.0, -0.86, -0.51, ELEMENT_GRASS, 0, 34.0, 13.5),
        Projectile(940.0, 490.0, 0.88, -0.47, ELEMENT_WATER, 1, 34.0, 13.5),
    ]
    eng.enemy_bullets = [EnemyProjectile(440.0, 360.0, 0.8, 0.6, ELEMENT_WATER)]
    eng.visual_events.append({"type": "callout", "x": 350.0, "y": 250.0, "text": "SHIELD BREAK!", "color": (56, 189, 248)})
    env.overlay_info = {"order": [0, 1], "attention_links": [(0, 1, 0.78)]}
    eng.telemetry["score"] = 1850.0
    eng.telemetry["kills"] = 16.0
    eng.wave_index = 2
    eng.step_count = 890
    frame = env.render()
    _save_frame(frame, "stage3_coop_boss_overlay.png")
    env.close()


def capture_stage4() -> None:
    env = DotShotParallelEnv(stage=4, render_mode="rgb_array")
    env.reset(seed=40)
    eng = env.engine
    eng.core.hp = 460.0

    # Vanguard frontlining with Aegis Shield active
    vg = eng.agents[0]
    vg.x, vg.y = 560.0, 480.0
    vg.aim_dx, vg.aim_dy = -1.0, 0.0
    vg.shield_active_timer = 65
    vg.last_weapon = ELEMENT_FIRE

    # Striker flanking boss
    st = eng.agents[1]
    st.x, st.y = 840.0, 360.0
    st.aim_dx, st.aim_dy = 0.82, -0.57
    st.last_weapon = ELEMENT_WATER

    # Support emitting Repair & Mark pulse near Core
    sp = eng.agents[2]
    sp.x, sp.y = 720.0, 610.0
    sp.aim_dx, sp.aim_dy = 0.0, -1.0
    sp.repair_pulse_timer = 12
    sp.last_weapon = ELEMENT_GRASS

    boss = Enemy(uid=401, x=1040.0, y=230.0, enemy_type=ELEMENT_FIRE, is_boss=True)
    boss.shield_hp = 0.0
    boss.shield_broken_timer = 110
    boss.marked_timer = 120
    boss.hp = 140.0

    e2 = Enemy(uid=402, x=320.0, y=470.0, enemy_type=ELEMENT_GRASS)
    e2.marked_timer = 120
    eng.enemies = [boss, e2, Enemy(uid=403, x=410.0, y=730.0, enemy_type=ELEMENT_WATER)]

    eng.bullets = [
        Projectile(920.0, 305.0, 0.82, -0.57, ELEMENT_WATER, 1, 46.0, 13.5),
        Projectile(450.0, 476.0, -1.0, 0.0, ELEMENT_FIRE, 0, 28.0, 11.5),
    ]
    eng.visual_events.append({"type": "callout", "x": 720.0, "y": 575.0, "text": "PULSE", "color": (34, 197, 94)})
    eng.visual_events.append({"type": "callout", "x": 1040.0, "y": 175.0, "text": "COMBO KILL!", "color": (34, 197, 94)})
    env.overlay_info = {"order": [0, 2, 1], "attention_links": [(0, 1, 0.85), (2, 1, 0.72)]}
    eng.telemetry["score"] = 3120.0
    eng.telemetry["kills"] = 27.0
    eng.wave_index = 3
    eng.step_count = 1320
    frame = env.render()
    _save_frame(frame, "stage4_heterogeneous_squad.png")
    env.close()


if __name__ == "__main__":
    capture_stage1()
    capture_stage2()
    capture_stage3()
    capture_stage4()
