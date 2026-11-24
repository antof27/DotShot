"""Optuna BOHB tuning (multivariate TPE + Hyperband), one study per (stage, learner).

Protocol (Phase 3 of the M3 performance fix):
* Every completed trial trains two seeds with identical hyperparameters and
  returns the mean of their final-policy returns on 32 ``val`` episodes. The
  final policy is scored, never the best checkpoint, and step 0 is never reported.
* Hyperband prunes on seed A's periodic ``val`` evaluations; seed B only runs
  for trials that reach ``max_resource``.
* Environment settings (``max_steps`` included) are the stage defaults, exactly
  as in the final runs; only learner hyperparameters are searched.

Usage:
    python -m benchmarks.official.bohb_tune --study s3_harl_mappo --trials 24
    python -m benchmarks.official.bohb_tune --export-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
import yaml

_SRC = Path(__file__).resolve().parents[2]
_REPO = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

SAMPLER_SEED = 42
TRIAL_SEEDS = (100, 101)
HARL_ALGOS = ("mappo", "happo", "hatrpo", "haa2c")
STUDIES = (
    "s1_sb3_ppo",
    "s2_sb3_ppo",
    *[f"s{s}_harl_{a}" for s in (3, 4) for a in HARL_ALGOS],
    "s3_mat",
    "s4_mat",
)
OPTUNA_DIR = _REPO / "runs" / "optuna"
CONFIG_DIR = _REPO / "configs" / "official"
SUMMARY_PATH = _REPO / "docs" / "results" / "optuna_summary.json"

# Fixed (not searched) settings shared by tuning and final runs.
SB3_FIXED = {"algo": "ppo", "num_envs": 8, "vec_env": "dummy", "cosine_lr": True, "vf_coef": 0.5, "max_grad_norm": 0.5}
MARL_FIXED = {"episode_length": 200, "n_rollout_threads": 32, "use_linear_lr_decay": True}


def _parse(study: str) -> tuple:
    stage, learner = study.split("_", 1)
    return int(stage[1:]), learner


# ----------------------------------------------------------------------- search spaces


def sb3_space(trial: optuna.Trial, stage: int) -> Dict[str, Any]:
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 10),
        "ent_coef": trial.suggest_float("ent_coef", 1e-3, 3e-2, log=True),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "gamma": trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
    }
    if stage == 2 and trial.suggest_categorical("warm_start", [False, True]):
        params["warm_start"] = True
    else:
        size = trial.suggest_categorical("net_arch_size", [128, 256])
        params["net_arch"] = [size, size]
    return params


def harl_space(trial: optuna.Trial, algo: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "critic_lr": trial.suggest_float("critic_lr", 1e-4, 1e-3, log=True),
        "critic_epoch": trial.suggest_int("critic_epoch", 3, 10),
        "critic_num_mini_batch": trial.suggest_categorical("critic_num_mini_batch", [1, 2, 4, 8]),
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 3e-2, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
    }
    if algo == "hatrpo":
        params["kl_threshold"] = trial.suggest_float("kl_threshold", 2e-3, 5e-2, log=True)
    else:
        params["lr"] = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        params["actor_num_mini_batch"] = trial.suggest_categorical("actor_num_mini_batch", [1, 2, 4, 8])
        if algo == "haa2c":
            params["a2c_epoch"] = trial.suggest_int("a2c_epoch", 1, 5)
        else:
            params["ppo_epoch"] = trial.suggest_int("ppo_epoch", 3, 10)
            params["clip_param"] = trial.suggest_categorical("clip_param", [0.1, 0.2, 0.3])
    hidden = trial.suggest_categorical("hidden_size", [128, 256])
    params["hidden_sizes"] = [hidden, hidden]
    if trial.suggest_categorical("warm_start", ["scratch", "bc"]) == "bc":
        params["warm_start"] = "bc"  # BC clones into an MLP actor
    else:
        params["use_recurrent_policy"] = trial.suggest_categorical("use_recurrent_policy", [False, True])
    return params


def mat_space(trial: optuna.Trial) -> Dict[str, Any]:
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 3e-2, log=True),
        "clip_param": trial.suggest_categorical("clip_param", [0.1, 0.2, 0.3]),
        "ppo_epoch": trial.suggest_int("ppo_epoch", 3, 10),
        "num_mini_batch": trial.suggest_categorical("num_mini_batch", [1, 2, 4, 8]),
        "gamma": trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
        "n_block": trial.suggest_int("n_block", 1, 3),
        "n_embd": trial.suggest_categorical("n_embd", [64, 128]),
        "n_head": trial.suggest_categorical("n_head", [1, 2, 4]),
    }
    if trial.suggest_categorical("warm_start", ["scratch", "bc"]) == "bc":
        params["warm_start"] = "bc"
    return params


def trial_config(study: str, trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    stage, learner = _parse(study)
    common = {
        "stage": stage,
        "total_steps": args.max_resource,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval,
        "n_test_episodes": args.n_final,
        "final_eval_split": "val",
    }
    if learner == "sb3_ppo":
        cfg = {**SB3_FIXED, **common, **sb3_space(trial, stage)}
        if cfg.pop("warm_start", False):
            ckpt, arch = stage1_reference(args.db_dir)
            if ckpt is None:
                raise optuna.TrialPruned("no Stage 1 checkpoint for warm start")
            # Final runs re-resolve init_model_path from the final Stage 1 runs.
            cfg.update(init_model_path=ckpt, net_arch=arch, reset_weapon_head=True, warm_start=True)
        return cfg
    if learner == "mat":
        return {"algo": "mat", **MARL_FIXED, **common, **mat_space(trial)}
    algo = learner[len("harl_"):]
    return {"algo": learner, **MARL_FIXED, **common, **harl_space(trial, algo)}


# ------------------------------------------------------------------------- objectives


def _train(cfg: Dict[str, Any], seed: int, device: str, run_dir: Path, trial: Optional[optuna.Trial]) -> Dict[str, Any]:
    algo = str(cfg["algo"])
    if algo.startswith("harl_"):
        from benchmarks.official.harl_runner import run_harl_experiment

        return run_harl_experiment(cfg, seed, device, str(run_dir), trial)
    if algo == "mat":
        from benchmarks.official.mat_runner import run_mat_experiment

        return run_mat_experiment(cfg, seed, device, str(run_dir), trial)
    from benchmarks.runners.run_single_agent import run_experiment

    return run_experiment(cfg, seed, device, run_dir_override=str(run_dir), optuna_trial=trial)


def make_objective(study: str, args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        cfg = trial_config(study, trial, args)
        trial.set_user_attr("config", cfg)
        root = OPTUNA_DIR / study / f"trial_{trial.number}"
        finals = []
        for k, seed in enumerate(TRIAL_SEEDS):
            result = _train(cfg, seed, args.device, root / f"seed_{seed}", trial if k == 0 else None)
            finals.append(result["final_return"])
            trial.set_user_attr(f"final_val_return_seed{seed}", result["final_return"])
        value = float(sum(finals) / len(finals))
        print(f"[{study}] trial {trial.number}: {value:.3f} (seeds {finals})", flush=True)
        return value

    return objective


# ----------------------------------------------------------------------------- studies


def _storage(db_dir: Path, study: str) -> str:
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(db_dir / f'{study}.db').resolve().as_posix()}"


def open_study(study: str, args: argparse.Namespace) -> optuna.Study:
    return optuna.create_study(
        study_name=study,
        storage=optuna.storages.RDBStorage(_storage(args.db_dir, study), engine_kwargs={"connect_args": {"timeout": 60}}),
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SAMPLER_SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=args.min_resource, max_resource=args.max_resource, reduction_factor=3
        ),
        direction="maximize",
        load_if_exists=True,
    )


def load_study(study: str, db_dir: Path) -> Optional[optuna.Study]:
    if not (db_dir / f"{study}.db").exists():
        return None
    try:
        return optuna.load_study(study_name=study, storage=_storage(db_dir, study))
    except KeyError:
        return None


def best_trial(study: Optional[optuna.Study]) -> Optional[optuna.trial.FrozenTrial]:
    try:
        return study.best_trial if study is not None else None
    except ValueError:
        return None


def stage1_reference(db_dir: Path) -> tuple:
    """Final checkpoint and trunk width of the best Stage 1 trial (seed A)."""
    best = best_trial(load_study("s1_sb3_ppo", db_dir))
    if best is None:
        return None, None
    ckpt = OPTUNA_DIR / "s1_sb3_ppo" / f"trial_{best.number}" / f"seed_{TRIAL_SEEDS[0]}" / "checkpoints" / "final.zip"
    size = int(best.params["net_arch_size"])
    return (str(ckpt), [size, size]) if ckpt.exists() else (None, None)


def _finished(study: optuna.Study) -> int:
    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(1 for t in study.trials if t.state in states)


def run_study(name: str, args: argparse.Namespace) -> None:
    study = open_study(name, args)
    if args.enqueue_from and not study.trials:
        src = best_trial(load_study(args.enqueue_from, args.db_dir))
        if src is not None:
            study.enqueue_trial(dict(src.params), skip_if_exists=True)
            print(f"[{name}] enqueued best params of {args.enqueue_from}", flush=True)
    objective = make_objective(name, args)
    while _finished(study) < args.trials:
        failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        if failed >= 5:
            raise RuntimeError(f"[{name}] {failed} failed trials; stopping")
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))


# ------------------------------------------------------------------------------ export


def _summary(study: optuna.Study) -> Dict[str, Any]:
    trials = sorted(study.trials, key=lambda t: t.number)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(complete, key=lambda t: t.value) if complete else None
    importances: Dict[str, float] = {}
    if len(complete) >= 4:
        try:
            importances = {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            print(f"[{study.study_name}] importances unavailable: {exc}")
    return {
        "best_value": float(best.value) if best else None,
        "best_trial": best.number if best else None,
        "best_params": dict(best.params) if best else {},
        "best_config": best.user_attrs.get("config") if best else None,
        "n_complete": len(complete),
        "n_pruned": sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED),
        "n_failed": sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL),
        "param_importances": importances,
        "trials": [
            {"number": t.number, "state": t.state.name, "value": t.value, "params": dict(t.params),
             "intermediate": {int(k): float(v) for k, v in t.intermediate_values.items()}}
            for t in trials
        ],
    }


def final_config(study: str, best_config: Dict[str, Any], final_steps: int) -> Dict[str, Any]:
    """Tuned config for the final runs: same hyperparameters, final budget, test-split scoring."""
    cfg = {k: v for k, v in best_config.items() if k not in ("final_eval_split", "init_model_path")}
    cfg.update(total_steps=final_steps, eval_freq=max(100_000, final_steps // 20), n_eval_episodes=16,
               n_test_episodes=32, exp_name=study)
    return cfg


def export(args: argparse.Namespace) -> None:
    entries = {}
    for name in STUDIES:
        study = load_study(name, args.db_dir)
        if study is not None:
            entries[name] = _summary(study)
    written = []
    for name, entry in entries.items():
        if not entry["best_config"]:
            continue
        stage, _ = _parse(name)
        steps = 2_000_000 if stage in (1, 2) else 4_000_000
        path = CONFIG_DIR / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Exported by benchmarks.official.bohb_tune from study {name} (trial {entry['best_trial']}).\n")
            yaml.safe_dump(final_config(name, entry["best_config"], steps), f, sort_keys=False)
        written.append(path.relative_to(_REPO).as_posix())
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "sampler": f"TPESampler(multivariate=True, seed={SAMPLER_SEED})",
            "pruner": f"HyperbandPruner(min_resource={args.min_resource}, max_resource={args.max_resource}, reduction_factor=3)",
            "trial_seeds": list(TRIAL_SEEDS),
            "objective": f"mean over seeds of the final policy's return on {args.n_final} val episodes",
        },
        "studies": entries,
        "exported_configs": written,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"exported {len(written)} config(s) and {SUMMARY_PATH.relative_to(_REPO).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=STUDIES)
    parser.add_argument("--trials", type=int, default=24, help="complete + pruned trials per study")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--min-resource", type=int, default=200_000)
    parser.add_argument("--max-resource", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=200_000)
    parser.add_argument("--n-eval", type=int, default=16, help="val episodes per pruning report")
    parser.add_argument("--n-final", type=int, default=32, help="val episodes scoring the final policy")
    parser.add_argument("--enqueue-from", default=None, help="study whose best params become the first trial")
    parser.add_argument("--db-dir", type=Path, default=OPTUNA_DIR)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.export_only:
        export(args)
        return
    if not args.study:
        parser.error("--study is required unless --export-only")
    run_study(args.study, args)
    export(args)


if __name__ == "__main__":
    main()
