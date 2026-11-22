"""Tests for all 9 modular MARL algorithms, QMIX monotonicity, policy loading, and rliable stats."""

import os
import tempfile
import numpy as np
import pytest
import torch

from benchmarks.eval.evaluate import compute_iqm, compute_rliable_stats
from benchmarks.marl.networks import QMIXMixer
from benchmarks.marl.policy_loader import load_marl_policy
from benchmarks.marl.train_marl import run_marl_experiment
from dotshot.envs.multi_agent import DotShotParallelEnv


def test_qmix_monotonicity_invariant():
    torch.manual_seed(0)
    mixer = QMIXMixer(num_agents=3, state_dim=64)
    state = torch.randn(8, 64)
    qs = torch.randn(8, 3, requires_grad=True)
    q_tot = mixer(qs, state)
    grads = torch.autograd.grad(q_tot.sum(), qs)[0]
    assert torch.all(grads >= -1e-7), f"QMIX violated monotonicity: min grad = {grads.min().item()}"


def test_rliable_stats_accuracy():
    vals = [10.0, 12.0, 14.0, 100.0]
    # Middle 50% is [12.0, 14.0] -> IQM = 13.0
    assert abs(compute_iqm(vals) - 13.0) < 1e-6
    stats = compute_rliable_stats(vals, n_boot=500, seed=42)
    assert stats["ci_low"] <= stats["iqm"] <= stats["ci_high"]
    assert stats["median"] == 13.0


@pytest.mark.parametrize(
    "algo,stage",
    [
        ("ippo", 3),
        ("vdn", 3),
        ("qmix", 3),
        ("mappo", 3),
        ("happo", 4),
        ("hatrpo", 4),
        ("mat", 3),
        ("pmat", 3),
        ("romat", 4),
    ],
)
def test_marl_algorithm_cuda_smoke_and_loader(algo, stage, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "algo": algo,
        "stage": stage,
        "exp_name": f"smoke_{algo}",
        "total_steps": 64,
        "num_envs": 2,
        "vec_env": "dummy",
        "n_steps": 16,
        "n_epochs": 2,
        "batch_size": 16,
        "chunk_len": 8,
        "buffer_capacity": 64,
        "hidden_dim": 64,
        "eval_freq": 64,
        "n_eval_episodes": 2,
        "max_steps": 40,
    }
    metrics = run_marl_experiment(cfg, seed=0, device_str=device)
    assert np.isfinite(metrics["eval/return_mean"])

    ckpt_path = tmp_path / "runs" / f"stage{stage}" / algo / f"smoke_{algo}" / "seed_0" / "checkpoints" / "final.pt"
    assert ckpt_path.exists()

    env = DotShotParallelEnv(stage=stage, max_steps=40)
    obs, _ = env.reset(seed=0)
    policy, loaded_algo = load_marl_policy(str(ckpt_path), env, device=device)
    assert loaded_algo == algo
    obs_arr = np.stack([obs[a] for a in env.possible_agents], axis=0)
    acts, overlay = policy.act_deterministic(obs_arr)
    assert acts.shape == (len(env.possible_agents), 4)
    if algo in ("mat", "pmat", "romat"):
        assert overlay is not None and "order" in overlay
    env.close()
