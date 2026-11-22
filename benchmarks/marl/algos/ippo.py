"""Decentralized Independent PPO (IPPO) entry point."""

from benchmarks.marl.algos.mappo import train_shared_ppo


def train_ippo(vec_env, cfg, logger, device, seed):
    return train_shared_ppo("ippo", vec_env, cfg, logger, device, seed)
