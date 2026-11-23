"""Continuous Box adapter for running the unmodified MAT code on DotShot Stages 3 and 4."""

import math

import numpy as np
from gymnasium import spaces

from dotshot.envs.multi_agent import DotShotParallelEnv

CONTINUOUS_ACT_DIM = 8


def continuous_to_multidiscrete(action: np.ndarray, aim_bins: int = 32) -> np.ndarray:
    """Map an 8-D continuous MAT action to DotShot's MultiDiscrete action."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    dx = 0 if a[0] < -0.33 else (2 if a[0] > 0.33 else 1)
    dy = 0 if a[1] < -0.33 else (2 if a[1] > 0.33 else 1)
    angle = math.atan2(float(a[3]), float(a[2])) % (2.0 * math.pi)
    aim_bin = int(np.floor((angle / (2.0 * math.pi)) * aim_bins + 0.5)) % aim_bins
    weapon = int(np.argmax(a[4:8]))
    return np.array([dx, dy, aim_bin, weapon], dtype=np.int64)


class DotShotContinuousMATEnv:
    """Expose a DotShot ParallelEnv as a MAT-style continuous multi-agent environment."""

    def __init__(
        self,
        stage: int = 3,
        partial_obs: bool = True,
        reward_alpha: float = 0.5,
        max_steps: int | None = None,
        seed: int = 0,
    ):
        self.env = DotShotParallelEnv(
            stage=stage,
            partial_obs=partial_obs,
            reward_alpha=reward_alpha,
            max_steps=max_steps,
        )
        self.n_agents = len(self.env.possible_agents)
        self._seed = int(seed)
        self._needs_seed = True

        self.observation_space = [self.env.observation_space(a) for a in self.env.possible_agents]
        self.share_observation_space = [self.env.state_space() for _ in range(self.n_agents)]
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(CONTINUOUS_ACT_DIM,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

    def seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._needs_seed = True

    def reset(self):
        obs_dict, _ = self.env.reset(seed=self._seed if self._needs_seed else None)
        self._needs_seed = False
        obs = np.stack([obs_dict[a] for a in self.env.possible_agents], axis=0).astype(np.float32)
        share_obs = np.repeat(self.env.state()[None, :], self.n_agents, axis=0).astype(np.float32)
        return obs, share_obs

    def step(self, actions: np.ndarray):
        action_arr = np.asarray(actions, dtype=np.float32)
        act_dict = {
            name: continuous_to_multidiscrete(action_arr[i])
            for i, name in enumerate(self.env.possible_agents)
        }
        obs_dict, reward_dict, _, _, info_dict = self.env.step(act_dict)

        dones = np.full(self.n_agents, not self.env.agents, dtype=bool)
        obs = np.stack([obs_dict[a] for a in self.env.possible_agents], axis=0).astype(np.float32)
        share_obs = np.repeat(self.env.state()[None, :], self.n_agents, axis=0).astype(np.float32)
        rewards = np.array(
            [[reward_dict[a]] for a in self.env.possible_agents], dtype=np.float32
        )
        infos = [dict(info_dict[a]) for a in self.env.possible_agents]
        return obs, share_obs, rewards, dones, infos


class DummyMATVecEnv:
    """Synchronous in-process vector env so MAT never pays Windows IPC costs."""

    def __init__(self, envs):
        self.envs = list(envs)
        self.n_rollout_threads = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.share_observation_space = self.envs[0].share_observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs_list, share_list = [], []
        for env in self.envs:
            obs, share_obs = env.reset()
            obs_list.append(obs)
            share_list.append(share_obs)
        return np.stack(obs_list), np.stack(share_list), None

    def step(self, actions):
        action_arr = np.asarray(actions, dtype=np.float32)
        obs_list, share_list, rewards_list, dones_list, infos_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, share_obs, rewards, dones, infos = env.step(action_arr[i])
            if np.all(dones):
                for info in infos:
                    info["episode_done"] = True
                obs, share_obs = env.reset()
            obs_list.append(obs)
            share_list.append(share_obs)
            rewards_list.append(rewards)
            dones_list.append(dones)
            infos_list.append(infos)
        return (
            np.stack(obs_list),
            np.stack(share_list),
            np.stack(rewards_list),
            np.stack(dones_list),
            infos_list,
            None,
        )

    def close(self) -> None:
        for env in self.envs:
            env.env.close()
