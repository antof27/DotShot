"""DotShot v2.2 environment tests: projectile observations, shaping, respawn and reward ordering."""

from dataclasses import replace

import numpy as np
import pytest

import dotshot.core.config as C
from dotshot.core.config import ELEMENT_FIRE, ELEMENT_WATER, ELEMENT_WEAKNESS
from dotshot.core.engine import OBS_DIM, PROJ_FEATS, DotShotEngine
from dotshot.core.entities import Enemy, EnemyProjectile
from benchmarks.eval.marl_eval import evaluate_marl
from benchmarks.runners.run_marl_baselines import POLICIES, aim_bin

PROJ_START = 129  # projectile block follows the v2.1 layout


def _quiet_engine(stage=3):
    e = DotShotEngine(stage=stage)
    e.reset(seed=0)
    e.enemies, e.enemy_bullets, e.bullets = [], [], []
    e.spawn_timer = 10**6
    return e


def test_incoming_projectile_is_observed_and_receding_one_is_not():
    e = _quiet_engine()
    a = e.agents[0]
    e.enemy_bullets = [EnemyProjectile(a.x + 200.0, a.y, -1.0, 0.0, ELEMENT_FIRE)]  # flying at the agent
    obs = e.get_local_obs(0)
    block = obs[PROJ_START:PROJ_START + PROJ_FEATS]
    assert obs.shape == (OBS_DIM,)
    assert block[6] == 1.0  # active flag
    assert block[0] > 0.0 and abs(block[1]) < 1e-6  # to the right, same height
    assert block[2] < 0.0  # moving left, toward the agent
    assert block[5] == pytest.approx(-1.0)  # miss distance ~0: direct hit course
    e.enemy_bullets = [EnemyProjectile(a.x + 200.0, a.y, 1.0, 0.0, ELEMENT_FIRE)]  # flying away
    assert e.get_local_obs(0)[PROJ_START + 6] == 0.0
    state = e.get_global_state()
    assert state.shape == (e.state_dim,)


def _fire_at(e, enemy_type, weapon):
    a = e.agents[0]
    enemy = Enemy(uid=999, x=a.x + 150.0, y=a.y, enemy_type=enemy_type)  # clear line of sight
    e.enemies = [enemy]
    act = np.array([1, 1, aim_bin(1.0, 0.0, e.cfg.aim_bins), weapon + 1])
    rest = {o.idx: np.array([1, 1, 0, 0]) for o in e.agents[1:]}
    _, _, _, info = e.step({0: act, **rest})
    return info["reward_components"]


def test_fire_time_element_credit_sign():
    right = _fire_at(_quiet_engine(), ELEMENT_FIRE, ELEMENT_WEAKNESS[ELEMENT_FIRE])
    wrong = _fire_at(_quiet_engine(), ELEMENT_FIRE, ELEMENT_FIRE)
    assert right["fire_element_credit"] > 0.0 > wrong["fire_element_credit"]
    assert right["aim_shaping"] > 0.0  # paid on the firing tick toward the lead point


def test_aim_shaping_not_paid_without_firing():
    e = _quiet_engine()
    a = e.agents[0]
    e.enemies = [Enemy(uid=5, x=a.x + 150.0, y=a.y, enemy_type=ELEMENT_WATER)]
    for _ in range(5):
        _, _, _, info = e.step({o.idx: np.array([1, 1, 0, 0]) for o in e.agents})
    assert info["reward_components"].get("aim_shaping", 0.0) == 0.0


def test_dead_agents_respawn_and_victory_is_terminal():
    e = _quiet_engine()
    a = e.agents[0]
    a.hp, a.alive, a.respawn_timer = 0.0, False, 3
    noop = {o.idx: np.array([1, 1, 0, 0]) for o in e.agents}
    for _ in range(3):
        _, term, _, _ = e.step(noop)
        assert not term
    assert a.alive and a.hp == a.max_hp
    for agent in e.agents:  # all dead no longer ends the episode
        agent.hp, agent.alive, agent.respawn_timer = 0.0, False, 50
    _, term, trunc, _ = e.step(noop)
    assert not term and not trunc
    e.step_count = e.cfg.max_steps - 1
    _, term, trunc, info = e.step(noop)
    assert term and not trunc and info["win"] == 1.0
    assert info["reward_components"]["team_win_bonus"] > 0.0


def test_reward_ordering_charge_below_losing_fight_below_win():
    """charge (suicide) < fighting but losing < winning, on the same Stage 3 seeds."""
    n = 4
    charge, _ = evaluate_marl({"stage": 3}, POLICIES["charge"](), n, split="val")
    win, eps = evaluate_marl({"stage": 3}, POLICIES["heuristic_calibrated"](), n, split="val")
    # Fight-and-lose: the same script with enemy fire made lethal enough that the Core falls.
    C.STAGE_CONFIGS[3], saved = replace(C.STAGE_CONFIGS[3], enemy_bullet_damage=60.0), C.STAGE_CONFIGS[3]
    try:
        lose, lose_eps = evaluate_marl({"stage": 3}, POLICIES["heuristic_calibrated"](), n, split="val")
    finally:
        C.STAGE_CONFIGS[3] = saved
    assert lose["eval/win_rate"] == 0.0 and lose["eval/kills_mean"] > 0.0
    assert any(ep["win"] == 1.0 for ep in eps)
    assert charge["eval/return_mean"] < lose["eval/return_mean"] < win["eval/return_mean"]


def test_fire_element_credit_is_zero_mean_for_random_weapon():
    """One right and two wrong elements must cancel, so the credit never discourages firing."""
    credits = [
        _fire_at(_quiet_engine(), ELEMENT_FIRE, w)["fire_element_credit"] for w in range(3)
    ]
    assert sum(credits) == pytest.approx(0.0)


def test_crosshair_reports_element_under_aim():
    e = _quiet_engine()
    a = e.agents[0]
    e.enemies = [Enemy(uid=7, x=a.x + 150.0, y=a.y, enemy_type=ELEMENT_WATER)]
    a.aim_dx, a.aim_dy = 1.0, 0.0
    block = e.get_local_obs(0)[-4:]
    assert block[ELEMENT_WATER] == 1.0 and block[3] == 1.0 and block.sum() == 2.0
    a.aim_dx, a.aim_dy = -1.0, 0.0  # aiming away
    assert e.get_local_obs(0)[-4:].sum() == 0.0


def test_action_repeat_equals_repeated_ticks():
    """action_repeat=k holds the action for k ticks and sums the rewards."""
    one = DotShotEngine(stage=3, env_overrides={"action_repeat": 1})
    four = DotShotEngine(stage=3, env_overrides={"action_repeat": 4})
    one.reset(seed=3)
    four.reset(seed=3)
    act = {0: np.array([2, 1, 5, 2]), 1: np.array([0, 2, 20, 3])}
    total = {0: 0.0, 1: 0.0}
    for _ in range(4):
        r, _, _, _ = one.step(act)
        total = {k: total[k] + r[k] for k in total}
    r4, _, _, info = four.step(act)
    assert four.step_count == one.step_count == 4
    assert r4 == pytest.approx(total)
    assert 0.0 <= info["hold_frac"] <= 1.0 and info["move_frac"] == 1.0


def test_guard_potential_telescopes():
    """Potential-based shaping sums to -Phi(s0) over an episode when gamma = 1 (no exploitable loop)."""
    e = DotShotEngine(stage=3, env_overrides={"guard_potential_coef": 10.0, "shaping_gamma": 1.0})
    e.reset(seed=4)
    phi0 = e._guard_potential()
    rng = np.random.default_rng(0)
    while True:
        _, term, trunc, info = e.step({a.idx: rng.integers(0, [3, 3, 32, 4]) for a in e.agents})
        if term or trunc:
            break
    assert info["reward_components"]["guard_potential"] == pytest.approx(-phi0, abs=1e-9)
