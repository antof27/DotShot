"""MultiDiscrete adapter for running upstream MAT on DotShot Stages 3 and 4."""

from typing import Optional

import numpy as np

from dotshot.envs.multi_agent import DotShotParallelEnv


class DotShotMATEnv:
    """Expose a DotShot ParallelEnv in MAT's list-of-agents format.

    Actions are the native ``MultiDiscrete([3, 3, 32, 4])`` integers. ``infos``
    carry ``bad_transition`` (time-limit truncation) and ``agent_alive``.
    """

    def __init__(
        self,
        stage: int = 3,
        partial_obs: Optional[bool] = None,
        reward_alpha: Optional[float] = None,
        max_steps: Optional[int] = None,
        seed: int = 0,
        reward_mode: Optional[str] = None,
    ):
        self.env = DotShotParallelEnv(
            stage=stage,
            partial_obs=partial_obs,
            reward_alpha=reward_alpha,
            max_steps=max_steps,
            reward_mode=reward_mode,
        )
        self.n_agents = len(self.env.possible_agents)
        self._seed = int(seed)
        self._needs_seed = True

        self.observation_space = [self.env.observation_space(a) for a in self.env.possible_agents]
        self.share_observation_space = [self.env.state_space() for _ in range(self.n_agents)]
        self.action_space = [self.env.action_space(a) for a in self.env.possible_agents]

    def _share_obs(self) -> np.ndarray:
        return np.repeat(self.env.state()[None, :], self.n_agents, axis=0).astype(np.float32)

    def reset(self):
        obs_dict, _ = self.env.reset(seed=self._seed if self._needs_seed else None)
        self._needs_seed = False
        obs = np.stack([obs_dict[a] for a in self.env.possible_agents], axis=0).astype(np.float32)
        return obs, self._share_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).astype(np.int64)
        act_dict = {name: actions[i] for i, name in enumerate(self.env.possible_agents)}
        obs_dict, reward_dict, terms, truncs, info_dict = self.env.step(act_dict)

        done = not self.env.agents
        bad_transition = bool(any(truncs.values()) and not any(terms.values()))
        dones = np.full(self.n_agents, done, dtype=bool)
        obs = np.stack([obs_dict[a] for a in self.env.possible_agents], axis=0).astype(np.float32)
        rewards = np.array([[reward_dict[a]] for a in self.env.possible_agents], dtype=np.float32)
        infos = [
            {**info_dict[a], "bad_transition": bad_transition} for a in self.env.possible_agents
        ]
        return obs, self._share_obs(), rewards, dones, infos


class DummyMATVecEnv:
    """Synchronous in-process vector env so MAT never pays Windows IPC costs.

    On episode end the terminal observations are kept in ``infos[i][0]`` as
    ``terminal_obs`` / ``terminal_share_obs`` before the env is reset, so the
    runner can bootstrap truncated episodes.
    """

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
        obs_list, share_list, rewards_list, dones_list, infos_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, share_obs, rewards, dones, infos = env.step(actions[i])
            if np.all(dones):
                infos[0]["terminal_obs"] = obs
                infos[0]["terminal_share_obs"] = share_obs
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
