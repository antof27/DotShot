"""Transfer runs must depend on the run seed, not on the seed stored in the checkpoint."""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from benchmarks.runners.run_single_agent import make_env_fn, run_experiment


def test_transfer_runs_differ_across_seeds(tmp_path):
    env = DummyVecEnv([make_env_fn(2, 0, 0, 20)])
    teacher = PPO("MlpPolicy", env, n_steps=32, batch_size=32, n_epochs=1, seed=7, device="cpu",
                  policy_kwargs={"net_arch": [16]})
    teacher.save(str(tmp_path / "teacher.zip"))

    cfg = {"algo": "ppo", "stage": 2, "total_steps": 32, "num_envs": 1, "max_steps": 20,
           "n_eval_episodes": 1, "n_test_episodes": 1, "eval_freq": 10_000,
           "init_model_path": str(tmp_path / "teacher.zip"), "reset_weapon_head": True}
    params = []
    for seed in (0, 1):
        run_dir = tmp_path / f"seed_{seed}"
        run_experiment(cfg, seed, "cpu", run_dir_override=str(run_dir))
        model = PPO.load(str(run_dir / "checkpoints" / "final.zip"), device="cpu")
        params.append(torch.cat([p.flatten() for p in model.policy.parameters()]))
    assert not torch.allclose(params[0], params[1])
