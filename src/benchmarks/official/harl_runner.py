"""Thin adapter that trains HARL algorithms on DotShot Stages 3 and 4.

``third_party/HARL`` stays unmodified. Two upstream behaviours are patched here
for DotShot's ``MultiDiscrete([3, 3, 32, 4])`` actions:

* PPO ratio (finding A): HARL stores the joint log-prob in all 4 buffer columns,
  so ``action_aggregation="prod"`` computes r**4. This runner defaults to
  ``"mean"``, which recovers the true joint ratio r.
* Entropy (finding B): upstream builds a (B, B) tensor per branch; the patched
  ``ACTLayer.evaluate_actions`` averages each branch entropy over active rows.
"""

import json
import time
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
from harl.models.base.act import ACTLayer
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

from benchmarks.eval.marl_eval import evaluate_marl
from benchmarks.official.harl_dotshot_env import DotShotHARLEnv
from benchmarks.utils.logger import ExperimentLogger
from benchmarks.utils.seeding import configure_torch, seed_everything

_BASE_ALGOS = ("mappo", "happo", "hatrpo", "haa2c")


# ------------------------------------------------------------ entropy patch (B)

_ORIGINAL_ACT_EVALUATE = ACTLayer.evaluate_actions


def _act_evaluate_actions(self, x, action, available_actions=None, active_masks=None):
    """Upstream ``ACTLayer.evaluate_actions`` with a masked MultiDiscrete entropy.

    Each branch entropy is averaged over active rows, then branches are summed, which
    equals upstream's value when every row is active (so ``entropy_coef`` keeps its scale).
    """
    if not self.multidiscrete_action:
        return _ORIGINAL_ACT_EVALUATE(self, x, action, available_actions, active_masks)
    log_probs, entropies = [], []
    for action_out, act in zip(self.action_outs, torch.transpose(action, 0, 1)):
        dist = action_out(x)
        log_probs.append(dist.log_probs(act.unsqueeze(-1)))
        ent = dist.entropy()
        if active_masks is None:
            entropies.append(ent.mean())
        else:
            mask = active_masks.reshape(-1)
            entropies.append((ent * mask).sum() / mask.sum().clamp(min=1.0))
    return torch.cat(log_probs, dim=-1).sum(dim=-1, keepdim=True), torch.stack(entropies).sum(), None


ACTLayer.evaluate_actions = _act_evaluate_actions


# --------------------------------------------------------- HATRPO MultiDiscrete


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
# remove once upstream HATRPO supports MultiDiscrete directly. The KL is HARL's
# kl_approx over the 42 concatenated log-probs, the same approximation upstream
# applies to Discrete actions.


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

    def train_step(self, algo: str) -> Dict[str, float]:
        """One upstream update; returns the mean of HARL's train infos."""
        if algo == "mappo":
            actor_infos, critic_info = OnPolicyMARunner.train(self)
        else:
            actor_infos, critic_info = OnPolicyHARunner.train(self)
        out = {
            f"train/{k}": float(np.mean([_scalar(info[k]) for info in actor_infos]))
            for k in actor_infos[0]
        }
        out.update({f"train/{k}": _scalar(v) for k, v in critic_info.items()})
        return out


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return float(np.mean(value))


class HARLEvalPolicy:
    """Deterministic per-agent HARL actors behind the ``marl_eval`` policy interface."""

    def __init__(self, runner: _DotShotHARLRunner):
        self.runner = runner

    def reset(self, env) -> None:
        r = self.runner
        r.prep_rollout()
        self.rnn = np.zeros((r.num_agents, r.recurrent_n, r.rnn_hidden_size), dtype=np.float32)
        self.masks = np.ones((1, 1), dtype=np.float32)

    @torch.no_grad()
    def act(self, env, obs) -> Dict[str, np.ndarray]:
        out = {}
        for i, name in enumerate(env.possible_agents):
            action, rnn = self.runner.actor[i].act(
                obs[name][None], self.rnn[i : i + 1], self.masks, deterministic=True
            )
            out[name] = action[0].cpu().numpy()
            self.rnn[i] = rnn[0].cpu().numpy()
        return out


def _save_checkpoint(runner: _DotShotHARLRunner, ckpt_dir: Path, meta: Dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runner.save_dir = str(ckpt_dir)
    runner.save()
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def build_harl_args(cfg: Dict[str, Any], algo: str) -> Dict[str, Any]:
    """Resolve the YAML config into HARL's model/algo/train argument dicts."""
    hidden_sizes = [int(size) for size in cfg.get("hidden_sizes", [128, 128])]
    model_args = {
        "hidden_sizes": hidden_sizes,
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": bool(cfg.get("use_recurrent_policy", False)),
        "recurrent_n": 1,
        "data_chunk_length": int(cfg.get("data_chunk_length", 10)),
        "lr": float(cfg.get("lr", 5e-4)),
        "critic_lr": float(cfg.get("critic_lr", 5e-4)),
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "std_x_coef": 1.0,
        "std_y_coef": 0.5,
    }
    train_args = {
        "episode_length": int(cfg.get("episode_length", 200)),
        "n_rollout_threads": int(cfg.get("n_rollout_threads", 32)),
        "use_valuenorm": bool(cfg.get("use_valuenorm", True)),
        "use_linear_lr_decay": bool(cfg.get("use_linear_lr_decay", True)),
        "use_proper_time_limits": True,
        "num_env_steps": int(cfg.get("total_steps", 200_000)),
        "model_dir": None,
    }
    algo_args = {
        "ppo_epoch": int(cfg.get("ppo_epoch", 5)),
        "a2c_epoch": int(cfg.get("a2c_epoch", 5)),
        "critic_epoch": int(cfg.get("critic_epoch", 5)),
        "use_clipped_value_loss": True,
        "clip_param": float(cfg.get("clip_param", 0.2)),
        "actor_num_mini_batch": int(cfg.get("actor_num_mini_batch", 4)),
        "critic_num_mini_batch": int(cfg.get("critic_num_mini_batch", 4)),
        "entropy_coef": float(cfg.get("entropy_coef", 0.01)),
        "value_loss_coef": 1.0,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
        "use_gae": True,
        "gamma": float(cfg.get("gamma", 0.99)),
        "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
        "use_huber_loss": True,
        "use_policy_active_masks": True,
        "huber_delta": 10.0,
        "action_aggregation": str(cfg.get("action_aggregation", "mean")),
        "share_param": bool(cfg.get("share_param", algo == "mappo")),
        "fixed_order": algo == "mappo",
        "kl_threshold": float(cfg.get("kl_threshold", 0.01)),
        "ls_step": 10,
        "accept_ratio": 0.5,
        "backtrack_coeff": 0.8,
    }
    return {"model": model_args, "algo": algo_args, "train": train_args}


def run_harl_experiment(
    config: dict,
    seed: int,
    device: str = "cuda",
    run_dir_override: Optional[str] = None,
    optuna_trial: object | None = None,
) -> dict:
    """Train one HARL algorithm on DotShot Stage 3 or 4.

    Periodic evaluations use the ``val`` seed split (curves, best checkpoint, pruning).
    The final policy is scored once on ``final_eval_split`` (``test`` by default,
    ``val`` for tuning) with ``n_test_episodes`` episodes, logged as ``<split>/...``.
    """
    cfg = dict(config or {})
    algo = _base_algo_name(cfg.get("algo", "harl_mappo"))
    stage = int(cfg.get("stage", 3))
    state_type = str(cfg.get("state_type", "EP"))
    env_kwargs = {
        "stage": stage,
        "partial_obs": cfg.get("partial_obs"),
        "reward_alpha": cfg.get("reward_alpha"),
        "max_steps": cfg.get("max_steps"),
        "reward_mode": cfg.get("reward_mode"),
        "env_overrides": cfg.get("env_overrides"),
    }
    total_steps = int(cfg.get("total_steps", 200_000))
    n_eval_episodes = int(cfg.get("n_eval_episodes", 16))
    n_test_episodes = int(cfg.get("n_test_episodes", 32))
    final_split = str(cfg.get("final_eval_split", "test"))
    eval_freq = int(cfg.get("eval_freq", 100_000))
    algo_args = build_harl_args(cfg, algo)
    episode_length = algo_args["train"]["episode_length"]
    n_threads = algo_args["train"]["n_rollout_threads"]
    # shortcut: HARL's ShareDummyVecEnv drops the batch axis for a single env;
    # require two or more until that wrapper is replaced.
    if n_threads < 2:
        raise ValueError("HARL ShareDummyVecEnv needs n_rollout_threads >= 2.")

    seed_everything(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    configure_torch(device)
    torch_device = torch.device(device)

    run_dir = Path(run_dir_override or f"runs/official/stage{stage}/{algo}/seed_{seed}")
    resolved_config = {
        **cfg,
        "algo": f"harl_{algo}",
        "seed": int(seed),
        "device": str(torch_device),
        "state_type": state_type,
        "harl_args": algo_args,
    }
    logger = ExperimentLogger(str(run_dir), resolved_config)

    def make_env(rank: int) -> DotShotHARLEnv:
        return DotShotHARLEnv(seed=seed + rank * 1000, state_type=state_type, **env_kwargs)

    envs: Optional[ShareDummyVecEnv] = None
    try:
        envs = ShareDummyVecEnv([lambda rank=rank: make_env(rank) for rank in range(n_threads)])
        runner = _DotShotHARLRunner(
            algo, algo_args, state_type, envs, num_agents=envs.n_agents, device=torch_device
        )
        if cfg.get("init_actor_dir"):
            load_actor_weights(runner, Path(cfg["init_actor_dir"]))
        if cfg.get("warm_start") == "bc":
            from benchmarks.official.behaviour_cloning import bc_harl_actor, load_dataset

            if algo_args["model"]["use_recurrent_policy"]:
                raise ValueError("BC warm start supports MLP actors only.")
            data = load_dataset(stage, int(cfg.get("bc_episodes", 40)), env_kwargs)
            logger.log_metrics(0, bc_harl_actor(runner, data, epochs=int(cfg.get("bc_epochs", 3))))
        meta = {
            "obs_dim": int(envs.observation_space[0].shape[0]),
            "act_dims": [int(dim) for dim in envs.action_space[0].nvec],
            "stage": stage,
            "algo": algo,
            "recurrent": algo_args["model"]["use_recurrent_policy"],
            "hidden_sizes": algo_args["model"]["hidden_sizes"],
            "share_param": algo_args["algo"]["share_param"],
            "state_type": state_type,
        }
        eval_policy = HARLEvalPolicy(runner)
        best_return = -float("inf")
        last_val: Dict[str, float] = {}

        def record_eval(step: int) -> None:
            nonlocal best_return, last_val
            metrics, _ = evaluate_marl(env_kwargs, eval_policy, n_eval_episodes, split="val")
            logger.log_metrics(step, metrics)
            last_val = metrics
            if metrics["eval/return_mean"] > best_return:
                best_return = metrics["eval/return_mean"]
                _save_checkpoint(runner, run_dir / "checkpoints" / "best", meta)
            if optuna_trial is not None and step > 0:
                optuna_trial.report(metrics["eval/return_mean"], step=step)
                if optuna_trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned()

        if optuna_trial is None:
            record_eval(0)

        runner.warmup()
        env_steps = 0
        update = 0
        updates = max(1, total_steps // (episode_length * n_threads))
        start = time.time()
        last_eval_step = 0
        while env_steps < total_steps:
            update += 1
            if algo_args["train"]["use_linear_lr_decay"]:
                for actor in {id(a): a for a in runner.actor}.values():
                    actor.lr_decay(update, updates)
                runner.critic.lr_decay(update, updates)

            runner.prep_rollout()
            for step in range(episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = runner.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions)
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
            train_metrics = runner.train_step(algo)
            runner.after_update()
            train_metrics["train/fps"] = env_steps / max(1e-9, time.time() - start)
            logger.log_metrics(env_steps, train_metrics)

            if eval_freq > 0 and env_steps - last_eval_step >= eval_freq:
                record_eval(env_steps)
                last_eval_step = env_steps

        if last_eval_step != env_steps:
            record_eval(env_steps)
        _save_checkpoint(runner, run_dir / "checkpoints" / "final", meta)

        prefix = "test" if final_split == "test" else "final_val"
        final_metrics, _ = evaluate_marl(
            env_kwargs, eval_policy, n_test_episodes, split=final_split, prefix=prefix
        )
        final_metrics["train/wall_time_s"] = time.time() - start
        logger.log_metrics(env_steps, final_metrics)
        return {
            "run_dir": str(run_dir),
            "best_val_return": float(best_return),
            "last_val_metrics": last_val,
            "final_metrics": final_metrics,
            "final_return": float(final_metrics[f"{prefix}/return_mean"]),
        }
    finally:
        if envs is not None:
            envs.close()
        logger.close()


def load_actor_weights(runner: _DotShotHARLRunner, ckpt_dir: Path) -> None:
    """Warm start: load ``actor_agent{i}.pt`` files (e.g. from behaviour cloning)."""
    for agent_id, actor in enumerate(runner.actor):
        path = ckpt_dir / f"actor_agent{agent_id}.pt"
        if not path.exists():
            path = ckpt_dir / "actor_agent0.pt"
        actor.actor.load_state_dict(torch.load(path, map_location=runner.device))
