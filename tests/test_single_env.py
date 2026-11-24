"""Tests for DotShot Game v2 single-agent Gymnasium environments (Stages 1-2)."""

import os
import subprocess
import sys
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import dotshot  # noqa: F401
from dotshot.core.arena import has_line_of_sight, resolve_circle_obstacles
from dotshot.core.config import ELEMENT_FIRE, ELEMENT_GRASS, ELEMENT_WATER, RectObstacle
from dotshot.core.engine import OBS_DIM
from dotshot.core.entities import Enemy
from dotshot.envs.single_agent import DotShotSingleEnv


def test_headless_no_pygame_import():
    code = (
        "import sys; from dotshot.envs.single_agent import DotShotSingleEnv; "
        "env = DotShotSingleEnv(stage=1); env.reset(seed=7); "
        "[env.step(env.action_space.sample()) for _ in range(10)]; "
        "env.close(); assert 'pygame' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"pygame leaked into headless run: {res.stderr}"


@pytest.mark.parametrize("stage,env_id", [(1, "DotShot-Stage1-v1"), (2, "DotShot-Stage2-v1")])
def test_gymnasium_env_checker(stage, env_id):
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset(seed=123)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert info["stage"] == stage
    env.close()


@pytest.mark.parametrize("stage", [1, 2])
def test_bitwise_seed_determinism(stage):
    actions = [np.array([2, 1, (t * 3) % 32, (t % 4)], dtype=np.int64) for t in range(40)]

    def rollout(seed):
        env = DotShotSingleEnv(stage=stage)
        obs0, _ = env.reset(seed=seed)
        traj = [obs0.copy()]
        rews = []
        for act in actions:
            obs, r, term, trunc, _ = env.step(act)
            traj.append(obs.copy())
            rews.append(r)
            if term or trunc:
                break
        env.close()
        return traj, rews

    traj_a, rews_a = rollout(999)
    traj_b, rews_b = rollout(999)
    assert len(traj_a) == len(traj_b)
    for oa, ob in zip(traj_a, traj_b):
        assert np.array_equal(oa, ob)
    assert rews_a == rews_b


def test_obstacle_collision_and_los_blocking():
    obstacles = [RectObstacle(200.0, 200.0, 100.0, 100.0, blocks_los=True)]
    nx, ny, hit = resolve_circle_obstacles(210.0, 250.0, 15.0, obstacles, 1440.0, 960.0)
    assert hit
    assert nx <= 185.0 + 1e-5
    assert not has_line_of_sight(150.0, 250.0, 350.0, 250.0, obstacles)
    assert has_line_of_sight(150.0, 150.0, 350.0, 150.0, obstacles)


def test_elemental_counter_damage_and_mismatch():
    env = DotShotSingleEnv(stage=2)
    env.reset(seed=42)
    env.engine.enemies.clear()
    env.engine.neutral_allies.clear()

    # Place Fire enemy directly to the right of agent (aim bin 0 -> +x)
    agent = env.engine.agents[0]
    fire_enemy = Enemy(uid=9001, x=agent.x + 80.0, y=agent.y, enemy_type=ELEMENT_FIRE)
    env.engine.enemies.append(fire_enemy)

    # Fire wrong element (Grass = action weapon 2)
    _, _, _, _, info_wrong = env.step(np.array([1, 1, 0, 2], dtype=np.int64))
    for _ in range(8):
        _, _, _, _, info_wrong = env.step(np.array([1, 1, 0, 0], dtype=np.int64))
    assert fire_enemy.hp == fire_enemy.max_hp
    assert info_wrong["hits_wrong"] >= 1.0

    # Reset cooldown and fire counter element (Water = action weapon 1)
    agent.fire_cooldown = 0
    _, _, _, _, info_right = env.step(np.array([1, 1, 0, 1], dtype=np.int64))
    for _ in range(8):
        _, _, _, _, info_right = env.step(np.array([1, 1, 0, 0], dtype=np.int64))
    assert fire_enemy.hp < fire_enemy.max_hp
    assert info_right["hits_correct"] >= 1.0
    env.close()


def test_offscreen_rgb_array_render():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = DotShotSingleEnv(stage=1, render_mode="rgb_array")
    env.reset(seed=1)
    env.step(np.array([2, 1, 0, 1], dtype=np.int64))
    frame = env.render()
    env.close()
    assert frame is not None
    assert frame.shape == (960, 1440, 3)
    assert frame.dtype == np.uint8
