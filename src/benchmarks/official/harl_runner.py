"""Thin adapter that trains unmodified HARL algorithms on DotShot Stages 3 and 4."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.algorithms.critics.v_critic import VCritic
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.common.valuenorm import ValueNorm
from harl.envs.env_wrappers import ShareDummyVecEnv
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

from benchmarks.official.harl_dotshot_env import DotShotHARLEnv
from benchmarks.utils.logger import ExperimentLogger
from benchmarks.utils.seeding import seed_everything
from dotshot.envs.multi_agent import DotShotParallelEnv

_BASE_ALGOS = ("mappo", "happo", "hatrpo", "haa2c")


class FixedCategorical:
    """Carry per-branch log-probabilities in the shape HARL's KL helper expects.

    HARL's TRPO utilities only read ``dist.logits`` and dispatch on the class
    name, so a factorized MultiDiscrete policy can reuse them by concatenating
    its per-branch log-probabilities under that name.
    """

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


_ORIGINAL_POLICY_EVALUATE = StochasticPolicy.evaluate_actions
_trpo_compat_enabled = False

# shortcut: relies on HARL's class-name KL dispatch and MultiDiscrete head layout;
# remove once upstream HATRPO supports MultiDiscrete directly.


def _evaluate_actions_with_factorized_logits(
    self,
    obs,
    rnn_states,
    action,
    masks,
    available_actions=None,
    active_masks=None,
):
    """Return upstream results plus a factorized distribution proxy for TRPO."""
    captured_logits: List[torch.Tensor] = []
    handles = []
    if getattr(self.act, "multidiscrete_action", False):
        handles = [
            head.register_forward_hook(
                lambda _module, _inputs, output: captured_logits.append(output.logits)
            )
            for head in self.act.action_outs
        ]
    try:
        log_probs, entropy, distribution = _ORIGINAL_POLICY_EVALUATE(
            self, obs, rnn_states, action, masks, available_actions, active_masks
        )
    finally:
        for handle in handles:
            handle.remove()

    if distribution is not None:
        return log_probs, entropy, distribution
    logits = torch.cat(
        [torch.log_softmax(branch, dim=-1) for branch in captured_logits], dim=-1
    )
    return log_probs, entropy, FixedCategorical(logits)


def _enable_trpo_multidiscrete_compat() -> None:
    global _trpo_compat_enabled
    if not _trpo_compat_enabled:
        StochasticPolicy.evaluate_actions = _evaluate_actions_with_factorized_logits
        _trpo_compat_enabled = True


class _MultiDiscreteHATRPO(HATRPO):
    """Upstream HATRPO with only its MultiDiscrete constructor guard widened.

    ``HATRPO.update`` and ``HATRPO.train`` are inherited unchanged; the proxy
    above supplies the distribution its KL helper requires.
    """

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        OnPolicyBase.__init__(self, args, obs_space, act_space, device)
        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]


def _base_algo_name(algo: str) -> str:
    name = str(algo).lower()
    if name.startswith("harl_"):
        name = name[len("harl_") :]
    if name not in _BASE_ALGOS:
        raise ValueError(f"Unsupported HARL algorithm: {algo}. Expected one of {_BASE_ALGOS}.")
    return name


class _DotShotHARLRunner(OnPolicyHARunner):
    """HARL runner that skips HARL's env/logger setup and keeps its RL math.

    The upstream ``train`` implementations only touch the attributes built here,
    so DotShot reuses the exact PPO/TRPO/factor updates and on-policy buffers.
    """

    def __init__(
        self,
        algo: str,
        algo_args: Dict[str, Any],
        state_type: str,
        envs: ShareDummyVecEnv,
        num_agents: int,
        device: torch.device,
    ):
        self.args = {"algo": f"harl_{algo}", "env": "dotshot", "exp_name": "official"}
        self.algo_args = algo_args
        self.env_args = {"state_type": state_type}
        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = state_type
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.device = device
        self.envs = envs
        self.num_agents = num_agents

        actor_args = {**algo_args["model"], **algo_args["algo"]}
        if algo == "hatrpo":
            _enable_trpo_multidiscrete_compat()
            actor_cls = _MultiDiscreteHATRPO
        else:
            actor_cls = ALGO_REGISTRY[algo]
        self.actor = []
        for agent_id in range(num_agents):
            if self.share_param and agent_id > 0:
                assert (
                    envs.observation_space[agent_id] == envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    envs.action_space[agent_id] == envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
                continue
            self.actor.append(
                actor_cls(
                    actor_args,
                    envs.observation_space[agent_id],
                    envs.action_space[agent_id],
                    device=device,
                )
            )

        self.actor_buffer = [
            OnPolicyActorBuffer(
                {**algo_args["train"], **algo_args["model"]},
                envs.observation_space[agent_id],
                envs.action_space[agent_id],
            )
            for agent_id in range(num_agents)
        ]

        share_obs_space = envs.share_observation_space[0]
        self.critic = VCritic(
            {**algo_args["model"], **algo_args["algo"]},
            share_obs_space,
            device=device,
        )
        critic_buffer_args = {
            **algo_args["train"],
            **algo_args["model"],
            **algo_args["algo"],
        }
        if state_type == "EP":
            self.critic_buffer = OnPolicyCriticBufferEP(critic_buffer_args, share_obs_space)
        elif state_type == "FP":
            self.critic_buffer = OnPolicyCriticBufferFP(
                critic_buffer_args, share_obs_space, num_agents
            )
        else:
            raise ValueError(f"state_type must be 'EP' or 'FP', got {state_type!r}.")

        self.value_normalizer = (
            ValueNorm(1, device=device)
            if algo_args["train"]["use_valuenorm"]
            else None
        )
        self.save_dir: Optional[str] = None

    @torch.no_grad()
    def evaluate(
        self, env: DotShotParallelEnv, n_episodes: int, seed: int
    ) -> Dict[str, float]:
        """Run deterministic episodes on a clean DotShot environment."""
        self.prep_rollout()
        returns: List[float] = []
        wins: List[float] = []
        core_hp: List[float] = []
        elem_acc: List[float] = []
        kills: List[float] = []
        boss_kills: List[float] = []
        shield_breaks: List[float] = []

        for episode in range(max(1, n_episodes)):
            obs, _ = env.reset(seed=seed + episode * 7919)
            rnn_states = np.zeros(
                (self.num_agents, self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((1, 1), dtype=np.float32)
            episode_return = 0.0
            last_info: Dict[str, Any] = {}

            while True:
                agent_actions: Dict[str, np.ndarray] = {}
                for agent_id, agent in enumerate(env.possible_agents):
                    action, new_rnn_state = self.actor[agent_id].act(
                        np.expand_dims(obs[agent], axis=0),
                        rnn_states[agent_id : agent_id + 1],
                        masks,
                        deterministic=True,
                    )
                    agent_actions[agent] = action[0].cpu().numpy()
                    rnn_states[agent_id] = new_rnn_state[0].cpu().numpy()

                obs, rewards, terminations, truncations, infos = env.step(agent_actions)
                episode_return += float(
                    np.mean([rewards.get(agent, 0.0) for agent in env.possible_agents])
                )
                last_info = infos[env.possible_agents[0]]
                if any(terminations.values()) or any(truncations.values()):
                    break

            returns.append(episode_return)
            wins.append(float(last_info.get("win", 0.0)))
            core_hp.append(float(last_info.get("core_hp_ratio", 1.0)))
            elem_acc.append(float(last_info.get("elemental_accuracy", 0.0)))
            kills.append(float(last_info.get("kills", 0.0)))
            boss_kills.append(float(last_info.get("boss_kills", 0.0)))
            shield_breaks.append(float(last_info.get("shield_breaks", 0.0)))

        return {
            "eval/return_mean": float(np.mean(returns)),
            "eval/return_std": float(np.std(returns)),
            "eval/win_rate": float(np.mean(wins)),
            "eval/core_hp_ratio": float(np.mean(core_hp)),
            "eval/elemental_accuracy": float(np.mean(elem_acc)),
            "eval/kills_mean": float(np.mean(kills)),
            "eval/boss_kills_mean": float(np.mean(boss_kills)),
            "eval/shield_breaks_mean": float(np.mean(shield_breaks)),
        }


def _save_checkpoint(runner: _DotShotHARLRunner, ckpt_dir: Path, meta: Dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runner.save_dir = str(ckpt_dir)
    runner.save()
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def run_harl_experiment(
    config: dict,
    seed: int,
    device: str = "cuda",
    run_dir_override: Optional[str] = None,
    optuna_trial: object | None = None,
) -> dict:
    """Train one unmodified HARL algorithm on DotShot Stage 3 or 4."""
    cfg = dict(config or {})
    algo = _base_algo_name(cfg.get("algo", "harl_mappo"))
    stage = int(cfg.get("stage", 3))
    partial_obs = bool(cfg.get("partial_obs", True))
    reward_alpha = float(cfg.get("reward_alpha", 0.5))
    max_steps = cfg.get("max_steps", 1500)
    max_steps = None if max_steps is None else int(max_steps)
    total_steps = int(cfg.get("total_steps", 200_000))
    episode_length = int(cfg.get("episode_length", 200))
    n_threads = int(cfg.get("n_rollout_threads", 8))
    n_eval_episodes = int(cfg.get("n_eval_episodes", 6))
    eval_freq = int(cfg.get("eval_freq", 50_000))
    # shortcut: HARL's ShareDummyVecEnv drops the batch axis for a single env;
    # require two or more until that wrapper is replaced.
    if n_threads < 2:
        raise ValueError(
            "HARL ShareDummyVecEnv drops the batch dimension for a single env; "
            "use n_rollout_threads >= 2."
        )

    hidden_sizes = [int(size) for size in cfg.get("hidden_sizes", [128, 128])]
    recurrent = bool(cfg.get("use_recurrent_policy", False))
    state_type = str(cfg.get("state_type", "FP"))
    share_param = bool(cfg.get("share_param", algo == "mappo"))
    use_valuenorm = bool(cfg.get("use_valuenorm", True))
    use_linear_lr_decay = bool(cfg.get("use_linear_lr_decay", False))
    lr = float(cfg.get("lr", 5e-4))
    critic_lr = float(cfg.get("critic_lr", 5e-4))
    ppo_epoch = int(cfg.get("ppo_epoch", 5))
    clip_param = float(cfg.get("clip_param", 0.2))
    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    seed_everything(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    model_args = {
        "hidden_sizes": hidden_sizes,
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": recurrent,
        "recurrent_n": 1,
        "data_chunk_length": 10,
        "lr": lr,
        "critic_lr": critic_lr,
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "std_x_coef": 1.0,
        "std_y_coef": 0.5,
    }
    train_args = {
        "episode_length": episode_length,
        "n_rollout_threads": n_threads,
        "use_valuenorm": use_valuenorm,
        "use_linear_lr_decay": use_linear_lr_decay,
        "use_proper_time_limits": True,
        "num_env_steps": total_steps,
        "model_dir": None,
    }
    algo_params = {
        "ppo_epoch": ppo_epoch,
        "a2c_epoch": ppo_epoch,
        "critic_epoch": 5,
        "use_clipped_value_loss": True,
        "clip_param": clip_param,
        "actor_num_mini_batch": 1,
        "critic_num_mini_batch": 1,
        "entropy_coef": entropy_coef,
        "value_loss_coef": 1.0,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
        "use_gae": True,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "use_huber_loss": True,
        "use_policy_active_masks": True,
        "huber_delta": 10.0,
        "action_aggregation": "prod",
        "share_param": share_param,
        "fixed_order": algo == "mappo",
        "kl_threshold": 0.01,
        "ls_step": 10,
        "accept_ratio": 0.5,
        "backtrack_coeff": 0.8,
    }
    algo_args = {"model": model_args, "algo": algo_params, "train": train_args}

    run_dir = Path(
        run_dir_override
        or f"runs/m3_official/stage{stage}/{algo}/seed_{seed}"
    )
    resolved_config = {
        "algo": f"harl_{algo}",
        "stage": stage,
        "seed": int(seed),
        "device": str(torch_device),
        "partial_obs": partial_obs,
        "reward_alpha": reward_alpha,
        "max_steps": max_steps,
        "total_steps": total_steps,
        "episode_length": episode_length,
        "n_rollout_threads": n_threads,
        "n_eval_episodes": n_eval_episodes,
        "eval_freq": eval_freq,
        "lr": lr,
        "critic_lr": critic_lr,
        "ppo_epoch": ppo_epoch,
        "clip_param": clip_param,
        "entropy_coef": entropy_coef,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "hidden_sizes": hidden_sizes,
        "use_recurrent_policy": recurrent,
        "share_param": share_param,
        "state_type": state_type,
        "use_valuenorm": use_valuenorm,
        "use_linear_lr_decay": use_linear_lr_decay,
    }
    logger = ExperimentLogger(str(run_dir), resolved_config)

    def make_env(rank: int) -> DotShotHARLEnv:
        return DotShotHARLEnv(
            stage=stage,
            partial_obs=partial_obs,
            reward_alpha=reward_alpha,
            max_steps=max_steps,
            seed=seed + rank * 1000,
        )

    envs: Optional[ShareDummyVecEnv] = None
    try:
        envs = ShareDummyVecEnv([lambda rank=rank: make_env(rank) for rank in range(n_threads)])
        runner = _DotShotHARLRunner(
            algo,
            algo_args,
            state_type,
            envs,
            num_agents=envs.n_agents,
            device=torch_device,
        )
        meta = {
            "obs_dim": int(envs.observation_space[0].shape[0]),
            "act_dims": [int(dim) for dim in envs.action_space[0].nvec],
            "stage": stage,
            "algo": algo,
            "recurrent": recurrent,
            "hidden_sizes": hidden_sizes,
            "share_param": share_param,
        }
        eval_seed = seed + 100_000
        best_return = -float("inf")
        final_metrics: Dict[str, float] = {}
        last_eval_step = 0

        def record_eval(step: int) -> None:
            nonlocal best_return, final_metrics
            eval_env = DotShotParallelEnv(
                stage=stage,
                partial_obs=partial_obs,
                reward_alpha=reward_alpha,
                max_steps=max_steps,
            )
            try:
                metrics = runner.evaluate(eval_env, n_eval_episodes, eval_seed)
            finally:
                eval_env.close()

            logger.log_metrics(step, metrics)
            final_metrics = metrics
            if metrics["eval/return_mean"] > best_return:
                best_return = metrics["eval/return_mean"]
                _save_checkpoint(runner, run_dir / "checkpoints" / "best", meta)
            if optuna_trial is not None:
                optuna_trial.report(metrics["eval/return_mean"], step=step)
                if optuna_trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()

        record_eval(0)

        env_steps = 0
        episode = 0
        episodes = max(1, total_steps // (episode_length * n_threads))
        while env_steps < total_steps:
            episode += 1
            if use_linear_lr_decay:
                if share_param:
                    runner.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(runner.num_agents):
                        runner.actor[agent_id].lr_decay(episode, episodes)
                runner.critic.lr_decay(episode, episodes)

            runner.prep_rollout()
            for step in range(episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    runner.collect(step)
                )
                obs, share_obs, rewards, dones, infos, available_actions = envs.step(
                    actions
                )
                runner.insert(
                    (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )
                )

            env_steps += episode_length * n_threads
            runner.compute()
            runner.prep_training()
            if algo == "mappo":
                OnPolicyMARunner.train(runner)
            else:
                OnPolicyHARunner.train(runner)
            runner.after_update()

            if eval_freq > 0 and env_steps - last_eval_step >= eval_freq:
                record_eval(env_steps)
                last_eval_step = env_steps

        if last_eval_step != env_steps:
            record_eval(env_steps)
        _save_checkpoint(runner, run_dir / "checkpoints" / "final", meta)

        return {
            "run_dir": str(run_dir),
            "best_return": float(best_return),
            "final_metrics": final_metrics,
        }
    finally:
        if envs is not None:
            envs.close()
        logger.close()
