"""Tests for the unmodified-upstream HARL and MAT adapters on DotShot Stages 3 and 4."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.official.mat_dotshot_env import DummyMATVecEnv, DotShotMATEnv
from dotshot.core.engine import OBS_DIM


def test_mat_env_and_smoke_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the official adapter smoke test")

    for stage in (3, 4):
        env = DotShotMATEnv(stage=stage, max_steps=40, seed=0)
        assert len(env.action_space) == env.n_agents
        assert all(list(space.nvec) == [3, 3, 32, 4] for space in env.action_space)
        assert all(space.shape == (OBS_DIM,) for space in env.observation_space)
        assert all(
            space.shape == env.share_observation_space[0].shape
            for space in env.share_observation_space
        )

        obs, share_obs = env.reset()
        assert obs.shape == (env.n_agents, OBS_DIM)
        assert share_obs.shape == (env.n_agents, env.share_observation_space[0].shape[0])
        obs, share_obs, rewards, dones, infos = env.step(np.ones((env.n_agents, 4), dtype=np.int64))
        assert obs.shape == (env.n_agents, OBS_DIM)
        assert rewards.shape == (env.n_agents, 1)
        assert rewards.dtype == np.float32
        assert dones.shape == (env.n_agents,)
        assert dones.dtype == bool
        assert len(infos) == env.n_agents

    # The dummy vector env must reset each thread as soon as its episode ends.
    vec = DummyMATVecEnv(
        [DotShotMATEnv(stage=3, max_steps=2, seed=1) for _ in range(2)]
    )
    obs, share_obs, available = vec.reset()
    assert available is None
    assert obs.shape == (2, vec.envs[0].n_agents, OBS_DIM)
    for _ in range(2):
        obs, share_obs, rewards, dones, infos, available = vec.step(
            np.ones((2, vec.envs[0].n_agents, 4), dtype=np.int64)
        )
        assert available is None
    assert dones.shape == (2, vec.envs[0].n_agents)
    assert bool(np.all(dones))
    assert vec.envs[0].env.engine.step_count == 0
    # v2.2: reaching max_steps on Stage 3 is a terminal victory, not a time-limit truncation.
    assert infos[0][0]["win"] == 1.0 and not infos[0][0]["bad_transition"]
    assert "terminal_obs" in infos[0][0]
    vec.close()

    from benchmarks.official.mat_runner import run_mat_experiment

    for stage in (3, 4):
        run_dir = tmp_path / f"mat_stage{stage}"
        config = {
            "stage": stage,
            "exp_name": f"smoke_mat_stage{stage}",
            "total_steps": 640,
            "episode_length": 80,
            "n_rollout_threads": 4,
            "n_eval_episodes": 2,
            "n_test_episodes": 2,
            "eval_freq": 320,
            "partial_obs": True,
            "reward_alpha": 0.5,
            "max_steps": 100,
            "n_block": 1,
            "n_embd": 32,
            "n_head": 1,
            "hidden_size": 32,
            "ppo_epoch": 1,
            "num_mini_batch": 1,
            "lr": 3e-4,
        }
        result = run_mat_experiment(
            config, seed=0, device="cuda", run_dir_override=str(run_dir)
        )
        assert Path(result["run_dir"]) == run_dir
        assert "best_val_return" in result
        assert "test/return_mean" in result["final_metrics"]

        metrics_path = run_dir / "metrics.jsonl"
        assert metrics_path.exists()
        records = [
            json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
        ]
        eval_records = [record for record in records if "eval/return_mean" in record]
        assert eval_records
        required = {
            "eval/return_mean",
            "eval/return_std",
            "eval/win_rate",
            "eval/core_hp_ratio",
            "eval/elemental_accuracy",
            "eval/kills_mean",
            "eval/boss_kills_mean",
            "eval/shield_breaks_mean",
        }
        for record in eval_records:
            assert required <= set(record)

        for split in ("best", "final"):
            ckpt_dir = run_dir / "checkpoints" / split
            assert (ckpt_dir / "transformer.pt").exists()
            meta = json.loads((ckpt_dir / "meta.json").read_text())
            assert meta["nvec"] == [3, 3, 32, 4]
            assert meta["stage"] == stage
            assert meta["obs_dim"] == OBS_DIM
            assert meta["n_agents"] > 0
            state_dict = torch.load(ckpt_dir / "transformer.pt", map_location="cpu")
            assert state_dict


@pytest.mark.parametrize(
    "algo,stage",
    [
        ("mappo", 3),
        ("happo", 4),
        ("hatrpo", 4),
        ("haa2c", 4),
    ],
)
def test_harl_env_spaces_and_smoke_cuda(algo, stage, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the official adapter smoke test")

    from benchmarks.official.harl_runner import run_harl_experiment

    run_dir = tmp_path / f"harl_{algo}_stage{stage}"
    config = {
        "algo": algo,
        "algorithm": algo,
        "algorithm_name": algo,
        "stage": stage,
        "exp_name": f"smoke_{algo}_stage{stage}",
        "total_steps": 640,
        "num_env_steps": 640,
        "episode_length": 80,
        "n_steps": 80,
        "n_rollout_threads": 4,
        "num_envs": 4,
        "eval_freq": 640,
        "eval_interval": 640,
        "n_eval_episodes": 2,
        "n_test_episodes": 2,
        "partial_obs": True,
        "reward_alpha": 0.5,
        "max_steps": 100,
    }
    result = run_harl_experiment(
        config, seed=0, device="cuda", run_dir_override=str(run_dir)
    )
    if isinstance(result, dict) and "run_dir" in result:
        run_dir = Path(result["run_dir"])

    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints" / "best" / "actor_agent0.pt").exists()
