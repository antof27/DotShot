"""Tests for DotShot Game v2 multi-agent PettingZoo ParallelEnv environments (Stages 3-4)."""

import os
import numpy as np
from pettingzoo.test import parallel_api_test
import pytest

from dotshot.core.config import ELEMENT_FIRE, ELEMENT_GRASS, ELEMENT_WATER
from dotshot.core.entities import Enemy
from dotshot.envs.multi_agent import DotShotParallelEnv


@pytest.mark.parametrize("stage", [3, 4])
def test_pettingzoo_parallel_api(stage):
    env = DotShotParallelEnv(stage=stage, max_steps=150)
    parallel_api_test(env, num_cycles=120)
    env.close()


@pytest.mark.parametrize("stage,num_agents", [(3, 2), (4, 3)])
def test_global_state_and_local_obs_contracts(stage, num_agents):
    env = DotShotParallelEnv(stage=stage)
    obs, infos = env.reset(seed=77)
    assert len(obs) == num_agents
    for name in env.possible_agents:
        assert obs[name].shape == (129,)
        assert obs[name].dtype == np.float32
        assert np.all(obs[name] >= -1.0) and np.all(obs[name] <= 1.0)

    state = env.state()
    assert state.shape == env.state_space().shape
    assert state.dtype == np.float32
    assert np.all(state >= -1.0) and np.all(state <= 1.0)
    env.close()


def test_stage3_boss_shield_break_mechanic():
    env = DotShotParallelEnv(stage=3)
    env.reset(seed=11)
    env.engine.enemies.clear()

    agent0 = env.engine.agents[0]
    boss = Enemy(
        uid=9999,
        x=agent0.x + 90.0,
        y=agent0.y,
        enemy_type=ELEMENT_WATER,
        is_boss=True,
    )
    boss.shield_hp = 20.0
    env.engine.enemies.append(boss)

    # Grass (weapon 2) counters Water boss shield
    actions = {a: np.array([1, 1, 0, 0], dtype=np.int64) for a in env.agents}
    actions["agent_0"] = np.array([1, 1, 0, 2], dtype=np.int64)
    _, _, _, _, infos = env.step(actions)
    for _ in range(10):
        if not env.agents:
            break
        actions = {a: np.array([1, 1, 0, 0], dtype=np.int64) for a in env.agents}
        _, _, _, _, infos = env.step(actions)

    assert boss.shield_hp == 0.0
    assert boss.shield_broken_timer > 0
    assert infos["agent_0"]["shield_breaks"] >= 1.0
    env.close()


def test_stage4_heterogeneous_roles_and_abilities():
    env = DotShotParallelEnv(stage=4)
    obs, infos = env.reset(seed=19)
    roles = [env.engine.agents[i].role_name for i in range(3)]
    assert roles == ["Vanguard", "Striker", "Support"]

    # Put Vanguard near an enemy and wound Core so both abilities trigger on action[3] == 0
    vanguard = env.engine.agents[0]
    support = env.engine.agents[2]
    env.engine.core.hp = 400.0
    enemy = Enemy(uid=8888, x=vanguard.x + 120.0, y=vanguard.y, enemy_type=ELEMENT_GRASS)
    env.engine.enemies.append(enemy)

    actions = {
        "agent_0": np.array([1, 1, 0, 0], dtype=np.int64),
        "agent_1": np.array([1, 1, 0, 0], dtype=np.int64),
        "agent_2": np.array([1, 1, 0, 0], dtype=np.int64),
    }
    env.step(actions)

    assert vanguard.shield_active_timer > 0
    assert support.repair_pulse_timer > 0
    assert env.engine.core.hp > 400.0
    assert env.engine.telemetry["marks_applied"] >= 1.0
    env.close()


def test_reward_alpha_mixing_and_offscreen_render():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env_ind = DotShotParallelEnv(stage=3, reward_alpha=0.0, render_mode="rgb_array")
    env_team = DotShotParallelEnv(stage=3, reward_alpha=1.0)
    env_ind.reset(seed=55)
    env_team.reset(seed=55)

    # Trigger Core damage event inside step so r_team reflects team penalty while r_ind does not
    enemy = Enemy(
        uid=7777,
        x=env_ind.engine.core.x + env_ind.engine.core.radius + 4.0,
        y=env_ind.engine.core.y,
        enemy_type=ELEMENT_FIRE,
    )
    enemy2 = Enemy(
        uid=7777,
        x=env_team.engine.core.x + env_team.engine.core.radius + 4.0,
        y=env_team.engine.core.y,
        enemy_type=ELEMENT_FIRE,
    )
    env_ind.engine.enemies = [enemy]
    env_team.engine.enemies = [enemy2]

    acts = {a: np.array([1, 1, 0, 0], dtype=np.int64) for a in env_ind.agents}
    _, r_ind, _, _, _ = env_ind.step(acts)
    _, r_team, _, _, _ = env_team.step(acts)

    assert r_team["agent_0"] == r_team["agent_1"]  # pure team signal (alpha=1.0) is identical across agents
    assert r_team["agent_0"] < r_ind["agent_0"]    # Core damage reduces team reward relative to pure individual reward

    frame = env_ind.render()
    assert frame.shape == (960, 1440, 3)
    env_ind.close()
    env_team.close()
