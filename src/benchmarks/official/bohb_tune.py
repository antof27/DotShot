"""Optuna BOHB tuning (multivariate TPE + Hyperband) for the DotShot official suite.

Each study trains real runs from the resolved stage YAML plus the sampled
hyperparameters, reports every evaluation to Optuna so Hyperband can stop weak
trials early, and finally exports the winning hyperparameters to
``configs/official/`` and ``docs/results/m3_bohb_summary.json``.

Usage:
    python -m benchmarks.official.bohb_tune --study all --trials 16 --device cuda
    python -m benchmarks.official.bohb_tune --export-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import optuna
import optuna.trial
import yaml

_SRC = Path(__file__).resolve().parents[2]
_REPO = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SEED = 42
STUDIES = ("sb3_ppo_stage1", "sb3_ppo_stage2", "harl_mappo_stage3", "official_mat_stage3")
HARL_ALGOS = ("mappo", "happo", "hatrpo", "haa2c")
TRIAL_ROOT = _REPO / "runs" / "bohb"
CONFIG_DIR = _REPO / "configs" / "official"
SUMMARY_PATH = _REPO / "docs" / "results" / "m3_bohb_summary.json"

SB3_KEYS = ("lr", "n_steps", "batch_size", "n_epochs", "ent_coef", "clip_range", "gamma", "gae_lambda")
HARL_KEYS = (
    "lr",
    "critic_lr",
    "ppo_epoch",
    "clip_param",
    "entropy_coef",
    "gamma",
    "gae_lambda",
    "use_recurrent_policy",
)
MAT_KEYS = ("lr", "ppo_epoch", "clip_param", "entropy_coef", "n_block", "n_embd", "n_head", "gamma")


# --------------------------------------------------------------------------- helpers


def _rel(path: str | Path) -> str:
    """Repo-relative POSIX path when possible, so summaries stay machine-portable."""
    try:
        return Path(path).resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _base_config(stage: int, algo: str) -> Dict[str, Any]:
    with open(_REPO / "configs" / f"stage{stage}_{algo}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _trial_dir(study_name: str, number: int) -> Path:
    return TRIAL_ROOT / study_name / f"trial_{number}"


def _finish(trial: optuna.Trial, study_name: str, result: Dict[str, Any]) -> float:
    value = float(result["best_return"])
    trial.set_user_attr("run_dir", _rel(result["run_dir"]))
    print(f"[{study_name}] trial {trial.number}: value={value:.3f} | {_rel(result['run_dir'])}")
    return value


def _storage(db: str | Path) -> str:
    return f"sqlite:///{Path(db).resolve().as_posix()}"


def _open_study(study_name: str, args: argparse.Namespace) -> optuna.Study:
    Path(args.db).resolve().parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=study_name,
        storage=_storage(args.db),
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=args.min_resource,
            max_resource=args.max_resource,
            reduction_factor=3,
        ),
        direction="maximize",
        load_if_exists=True,
    )


def _load_study(study_name: str, db: str | Path) -> Optional[optuna.Study]:
    try:
        return optuna.load_study(study_name=study_name, storage=_storage(db))
    except KeyError:
        return None


# ----------------------------------------------------------------------- search spaces


def _sb3_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "n_epochs": trial.suggest_categorical("n_epochs", [4, 6, 8, 10]),
        "ent_coef": trial.suggest_float("ent_coef", 0.002, 0.03, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.15, 0.2, 0.25]),
        "gamma": trial.suggest_categorical("gamma", [0.985, 0.99, 0.995]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.92, 0.95, 0.98]),
        "net_arch_size": trial.suggest_categorical("net_arch_size", [128, 256]),
    }


def _harl_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 8e-4, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 2e-4, 1e-3, log=True),
        "ppo_epoch": trial.suggest_int("ppo_epoch", 4, 10),
        "clip_param": trial.suggest_categorical("clip_param", [0.15, 0.2, 0.25]),
        "entropy_coef": trial.suggest_float("entropy_coef", 0.003, 0.03, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.985, 0.99, 0.995]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.92, 0.95, 0.98]),
        "use_recurrent_policy": trial.suggest_categorical("use_recurrent_policy", [False, True]),
        "hidden_size": trial.suggest_categorical("hidden_size", [128, 256]),
    }


def _mat_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 7e-4, log=True),
        "ppo_epoch": trial.suggest_int("ppo_epoch", 5, 15),
        "clip_param": trial.suggest_categorical("clip_param", [0.1, 0.2, 0.25]),
        "entropy_coef": trial.suggest_float("entropy_coef", 0.002, 0.03, log=True),
        "n_block": trial.suggest_categorical("n_block", [1, 2]),
        "n_embd": trial.suggest_categorical("n_embd", [64, 128]),
        "n_head": trial.suggest_categorical("n_head", [2, 4]),
        "gamma": trial.suggest_categorical("gamma", [0.985, 0.99, 0.995]),
    }


# -------------------------------------------------------------------------- objectives


def _sb3_objective(
    cfg_base: Dict[str, Any],
    study_name: str,
    args: argparse.Namespace,
    ckpt: Optional[str] = None,
    arch: Optional[list] = None,
) -> Callable[[optuna.Trial], float]:
    from benchmarks.runners.run_single_agent import run_experiment

    def objective(trial: optuna.Trial) -> float:
        params = _sb3_space(trial)
        size = int(params.pop("net_arch_size"))
        cfg = {
            **cfg_base,
            **params,
            "num_envs": 8,
            "vec_env": "subproc",
            "total_steps": args.max_resource,
            "eval_freq": args.eval_freq,
            "n_eval_episodes": args.n_eval,
            "net_arch": [size, size],
        }
        init_model_path = None
        if ckpt:
            # warm-started policies must keep the Stage 1 trunk; shortcut: the reused
            # run_experiment only re-applies lr/ent_coef/gamma/gae_lambda/n_epochs.
            if trial.suggest_categorical("warm_start", [False, True]):
                init_model_path = ckpt
                cfg["net_arch"] = list(arch)
        result = run_experiment(
            cfg,
            seed=SEED,
            device=args.device,
            steps_override=args.max_resource,
            run_dir_override=str(_trial_dir(study_name, trial.number)),
            optuna_trial=trial,
            init_model_path=init_model_path,
        )
        return _finish(trial, study_name, result)

    return objective


def _harl_objective(study_name: str, args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    from benchmarks.official.harl_runner import run_harl_experiment

    base = _base_config(3, "mappo")

    def objective(trial: optuna.Trial) -> float:
        params = _harl_space(trial)
        hidden = int(params.pop("hidden_size"))
        cfg = {
            "stage": 3,
            "algo": "harl_mappo",
            "partial_obs": bool(base.get("partial_obs", True)),
            "reward_alpha": float(base.get("reward_alpha", 0.65)),
            "episode_length": 200,
            "n_rollout_threads": 32,
            "total_steps": args.max_resource,
            "eval_freq": args.eval_freq,
            "n_eval_episodes": args.n_eval,
            "hidden_sizes": [hidden, hidden],
            **params,
        }
        result = run_harl_experiment(
            cfg,
            seed=SEED,
            device=args.device,
            run_dir_override=str(_trial_dir(study_name, trial.number)),
            optuna_trial=trial,
        )
        return _finish(trial, study_name, result)

    return objective


def _mat_objective(study_name: str, args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    from benchmarks.official.mat_runner import run_mat_experiment

    base = _base_config(3, "mat")

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "stage": 3,
            "algo": "mat",
            "partial_obs": bool(base.get("partial_obs", True)),
            "reward_alpha": float(base.get("reward_alpha", 0.65)),
            "episode_length": 200,
            "n_rollout_threads": 32,
            "total_steps": args.max_resource,
            "eval_freq": args.eval_freq,
            "n_eval_episodes": args.n_eval,
            **_mat_space(trial),
        }
        result = run_mat_experiment(
            cfg,
            seed=SEED,
            device=args.device,
            run_dir_override=str(_trial_dir(study_name, trial.number)),
            optuna_trial=trial,
        )
        return _finish(trial, study_name, result)

    return objective


# --------------------------------------------------------------------- stage 1 reuse


def _best_trial(study: Optional[optuna.Study]) -> Optional[optuna.trial.FrozenTrial]:
    try:
        return study.best_trial if study is not None else None
    except ValueError:  # study has no COMPLETE trial yet
        return None


def _stage1_reference(
    args: argparse.Namespace, studies: Dict[str, optuna.Study]
) -> Tuple[Optional[str], Optional[list]]:
    """Resolve the Stage 1 checkpoint and trunk width used by warm-start trials."""
    study = studies.get("sb3_ppo_stage1") or _load_study("sb3_ppo_stage1", args.db)
    arch = None
    ckpt = args.stage1_best_ckpt
    best = _best_trial(study)
    if best is not None:
        size = best.params.get("net_arch_size")
        arch = [int(size), int(size)] if size else None
        if not ckpt and best.user_attrs.get("run_dir"):
            ckpt = str(_REPO / best.user_attrs["run_dir"] / "checkpoints" / "best.zip")
    if ckpt and not Path(ckpt).exists():
        print(f"[sb3_ppo_stage2] Stage 1 checkpoint not found at {ckpt}; warm start disabled")
        ckpt = None
    if ckpt and arch is None:
        from stable_baselines3 import PPO

        model = PPO.load(ckpt, device="cpu")
        arch = [int(model.policy.mlp_extractor.latent_dim_pi)] * 2
    return (ckpt, arch) if ckpt and arch else (None, None)


# ------------------------------------------------------------------------- study run


def _finished(study: optuna.Study) -> int:
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(1 for trial in study.trials if trial.state in states)


def _run_study(
    study_name: str, study: optuna.Study, args: argparse.Namespace, studies: Dict[str, optuna.Study]
) -> None:
    if study_name == "sb3_ppo_stage1":
        objective = _sb3_objective(_base_config(1, "ppo"), study_name, args)
    elif study_name == "sb3_ppo_stage2":
        ckpt, arch = _stage1_reference(args, studies)
        objective = _sb3_objective(_base_config(2, "ppo"), study_name, args, ckpt, arch)
    elif study_name == "harl_mappo_stage3":
        objective = _harl_objective(study_name, args)
    else:
        objective = _mat_objective(study_name, args)

    while _finished(study) < args.trials:
        remaining = args.trials - _finished(study)
        print(f"[{study_name}] {_finished(study)} complete/pruned, running {remaining} trial(s)")
        study.optimize(objective, n_trials=remaining)


# ---------------------------------------------------------------------------- export


def _trial_value(trial: optuna.trial.FrozenTrial) -> Optional[float]:
    if trial.value is not None:
        return float(trial.value)
    if trial.intermediate_values:
        return float(max(trial.intermediate_values.values()))
    return None


def _summarize(study: optuna.Study) -> Dict[str, Any]:
    trials = sorted(study.trials, key=lambda t: t.number)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(complete, key=lambda t: t.value) if complete else None
    importances: Dict[str, float] = {}
    if len(complete) >= 2:
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception as exc:  # noqa: BLE001 - importance is diagnostic, never fatal
            print(f"[{study.study_name}] param importances unavailable: {exc}")
    return {
        "best_value": float(best.value) if best else None,
        "best_params": dict(best.params) if best else {},
        "best_trial": best.number if best else None,
        "best_run_dir": best.user_attrs.get("run_dir") if best else None,
        "n_complete": len(complete),
        "n_pruned": sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED),
        "param_importances": {k: float(v) for k, v in importances.items()},
        "trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": _trial_value(t),
                "params": dict(t.params),
                "run_dir": t.user_attrs.get("run_dir"),
            }
            for t in trials
        ],
    }


def _sb3_tuned(params: Dict[str, Any]) -> Dict[str, Any]:
    tuned = {k: params[k] for k in SB3_KEYS if k in params}
    if "net_arch_size" in params:
        tuned["net_arch"] = [int(params["net_arch_size"])] * 2
    return tuned


def _harl_cfg(algo: str, stage: int, params: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    tuned = {k: params[k] for k in HARL_KEYS if k in params}
    if "hidden_size" in params:
        tuned["hidden_sizes"] = [int(params["hidden_size"])] * 2
    return {
        "stage": stage,
        "algo": f"harl_{algo}",
        "exp_name": f"stage{stage}_harl_{algo}",
        "total_steps": int(base["total_steps"]),
        "episode_length": 200,
        "n_rollout_threads": 32,
        "partial_obs": bool(base.get("partial_obs", True)),
        "reward_alpha": float(base.get("reward_alpha", 0.65)),
        # only MAPPO keeps parameter sharing at Stage 3; HA algos and Stage 4 need separate actors
        "share_param": algo == "mappo" and stage == 3,
        "eval_freq": 100_000,
        "n_eval_episodes": 10,
        **tuned,
    }


def _mat_cfg(stage: int, params: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": stage,
        "algo": "mat",
        "exp_name": f"stage{stage}_official_mat",
        "total_steps": int(base["total_steps"]),
        "episode_length": 200,
        "n_rollout_threads": 32,
        "partial_obs": bool(base.get("partial_obs", True)),
        "reward_alpha": float(base.get("reward_alpha", 0.65)),
        "eval_freq": 100_000,
        "n_eval_episodes": 10,
        **{k: params[k] for k in MAT_KEYS if k in params},
    }


def _official_configs(
    bests: Dict[str, Dict[str, Any]],
    ckpt: Optional[str],
    arch: Optional[list],
) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    if bests.get("sb3_ppo_stage1"):
        configs["stage1_ppo.yaml"] = {
            **_base_config(1, "ppo"),
            **_sb3_tuned(bests["sb3_ppo_stage1"]),
        }

    stage2 = bests.get("sb3_ppo_stage2")
    if stage2:
        cfg_scratch = {**_base_config(2, "ppo"), **_sb3_tuned(stage2), "warm_start": False}
        configs["stage2_ppo.yaml"] = cfg_scratch
        cfg_transfer = {**_base_config(2, "ppo"), **_sb3_tuned(stage2), "warm_start": True}
        if ckpt and arch:
            cfg_transfer["init_model_path"] = _rel(ckpt)
            cfg_transfer["net_arch"] = list(arch)
        configs["stage2_ppo_transfer.yaml"] = cfg_transfer

    harl = bests.get("harl_mappo_stage3")
    if harl:
        for stage, base in ((3, _base_config(3, "mappo")), (4, _base_config(4, "happo"))):
            for algo in HARL_ALGOS:
                configs[f"stage{stage}_harl_{algo}.yaml"] = _harl_cfg(algo, stage, harl, base)

    mat = bests.get("official_mat_stage3")
    if mat:
        configs["stage3_official_mat.yaml"] = _mat_cfg(3, mat, _base_config(3, "mat"))
        configs["stage4_official_mat.yaml"] = _mat_cfg(4, mat, _base_config(4, "happo"))
    return configs


def _export(studies: Dict[str, optuna.Study], args: argparse.Namespace) -> None:
    entries: Dict[str, Any] = {}
    for name in STUDIES:
        study = studies.get(name) or _load_study(name, args.db)
        if study is not None:
            entries[name] = _summarize(study)

    bests = {name: e["best_params"] for name, e in entries.items() if e["best_value"] is not None}
    ckpt, arch = _stage1_reference(args, studies)
    written = []
    for filename, cfg in _official_configs(bests, ckpt, arch).items():
        path = CONFIG_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append(_rel(path))

    summary: Dict[str, Any] = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    summary["studies"] = {**summary.get("studies", {}), **entries}
    summary["exported_configs"] = sorted(set(summary.get("exported_configs", [])) | set(written))
    summary.update(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "db": _rel(args.db),
            "seed": SEED,
            "trials_per_study": args.trials,
            "resources": {
                "min_resource": args.min_resource,
                "max_resource": args.max_resource,
                "eval_freq": args.eval_freq,
            },
            "sampler": "TPESampler(multivariate=True, seed=42)",
            "pruner": (
                f"HyperbandPruner(min_resource={args.min_resource}, "
                f"max_resource={args.max_resource}, reduction_factor=3)"
            ),
        }
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported {len(written)} config(s) and {_rel(SUMMARY_PATH)}")


# ------------------------------------------------------------------------------ main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna BOHB tuning (multivariate TPE + Hyperband) for DotShot official benchmarks."
    )
    parser.add_argument("--study", choices=[*STUDIES, "all"], default="all")
    parser.add_argument("--trials", type=int, default=16, help="Complete+pruned trials per study")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--min-resource", type=int, default=50_000, help="Hyperband minimum steps")
    parser.add_argument("--max-resource", type=int, default=350_000, help="Steps per trial and Hyperband maximum")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Env steps between pruner reports")
    parser.add_argument("--n-eval", type=int, default=6, help="Episodes per evaluation")
    parser.add_argument("--stage1-best-ckpt", type=str, default=None, help="Stage 1 checkpoint for warm start")
    parser.add_argument("--db", type=str, default=str(_REPO / "runs" / "optuna_bohb.db"))
    parser.add_argument("--export-only", action="store_true", help="Re-export configs and summary from the DB")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("CUDA unavailable, falling back to CPU")
            args.device = "cpu"

    studies: Dict[str, optuna.Study] = {}
    if args.export_only:
        _export(studies, args)
        return

    requested = STUDIES if args.study == "all" else (args.study,)
    for study_name in requested:
        study = _open_study(study_name, args)
        studies[study_name] = study
        _run_study(study_name, study, args, studies)

    _export(studies, args)


if __name__ == "__main__":
    main()
