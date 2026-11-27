"""Slow end-to-end sanity check for the fixed HARL pipeline (``pytest -m slow``)."""

import json
import math

import pytest


@pytest.mark.slow
def test_harl_mappo_stage3_short_run(tmp_path):
    """60k steps of fixed HARL-MAPPO on Stage 3: finite losses, ratio near 1, full eval schema."""
    from benchmarks.official.harl_runner import run_harl_experiment

    result = run_harl_experiment(
        {"algo": "harl_mappo", "stage": 3, "total_steps": 64_000, "n_rollout_threads": 16,
         "episode_length": 200, "eval_freq": 32_000, "n_eval_episodes": 4, "n_test_episodes": 4},
        seed=0, device="cpu", run_dir_override=str(tmp_path / "run"),
    )
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if "train/policy_loss" in r]
    assert len(train) == 20
    for r in train:
        assert all(math.isfinite(v) for v in r.values())
        assert 0.9 < r["train/ratio"] < 1.1  # true PPO ratio (r**4 would drift much further)
    final = result["final_metrics"]
    for key in ("return_mean", "win_rate", "core_hp_ratio", "kills_mean", "episode_length", "rc_team_core_damage"):
        assert f"test/{key}" in final
    assert final["test/n_episodes"] == 4.0
