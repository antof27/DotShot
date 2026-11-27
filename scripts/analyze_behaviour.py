"""Qualitative behaviour statistics of a trained run or a scripted policy.

Plays held-out ``test`` episodes and records, at every decision and for every
live agent: its position, its distance to the Core, the weapon it chose and the
element of the enemy on its chosen aim ray, and whether it used its ability.
It also records the Core HP over time. The output JSON feeds
``scripts/plot_report_figures.py``.

    python scripts/analyze_behaviour.py --run runs/final/stage4/s4_harl_happo_bc/seed_0 --out docs/results/behaviour/s4_happo.json
    python scripts/analyze_behaviour.py --script heuristic_calibrated --stage 4 --out docs/results/behaviour/s4_script.json
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from dotshot.core.config import ROLE_NAMES
from dotshot.envs.multi_agent import DotShotParallelEnv
from dotshot.envs.single_agent import DotShotSingleEnv

GRID = (36, 24)  # heatmap bins over the 1440 x 960 arena (40 px cells)
TIME_BINS = 20


def _target_element(engine, agent, aim_bin: int):
    ang = aim_bin * (2.0 * np.pi / engine.cfg.aim_bins)
    uid = engine._find_ray_target_uid(agent, float(np.cos(ang)), float(-np.sin(ang)))
    target = next((e for e in engine.enemies if e.uid == uid), None) if uid is not None else None
    return None if target is None else int(target.enemy_type)


def analyze(policy, env_kwargs: dict, n_episodes: int) -> dict:
    from benchmarks.eval.marl_eval import episode_seeds

    stage = int(env_kwargs["stage"])
    multi = stage >= 3
    kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
    env = (DotShotParallelEnv if multi else DotShotSingleEnv)(**kwargs)
    engine = env.engine
    env.reset(seed=0)  # agents exist only after a reset
    roles = [ROLE_NAMES[a.role_id] for a in engine.agents]
    n = len(roles)
    heat = np.zeros((n, GRID[1], GRID[0]))
    confusion = np.zeros((n, 3, 4))  # [agent, target element, weapon action]
    abilities = np.zeros(n)
    dist = [[] for _ in range(n)]
    core_curves = np.zeros((n_episodes, TIME_BINS))
    episodes = []
    for ep, seed in enumerate(episode_seeds("test", n_episodes)):
        obs, _ = env.reset(seed=seed)
        policy.reset(env)
        core_trace = []
        while True:
            act = policy.act(env, obs)
            acts = act if multi else {0: act}
            for i, agent in enumerate(engine.agents):
                if not agent.alive:
                    continue
                a = np.asarray(acts[env.possible_agents[i]] if multi else acts[0], dtype=np.int64)
                gx = min(GRID[0] - 1, int(agent.x / engine.cfg.width * GRID[0]))
                gy = min(GRID[1] - 1, int(agent.y / engine.cfg.height * GRID[1]))
                heat[i, gy, gx] += 1
                if engine.core is not None:
                    dist[i].append(float(np.hypot(agent.x - engine.core.x, agent.y - engine.core.y)))
                elem = _target_element(engine, agent, int(a[2]))
                if elem is not None:
                    confusion[i, elem, int(a[3])] += 1
            ready = [a.ability_cooldown == 0 and a.alive for a in engine.agents]
            if multi:
                obs, _, terms, truncs, infos = env.step(act)
                done, info = any(terms.values()) or any(truncs.values()), infos[env.possible_agents[0]]
            else:
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
            for i, a in enumerate(engine.agents):
                abilities[i] += float(ready[i] and a.ability_cooldown > 0)
            if engine.core is not None:
                core_trace.append((engine.step_count, engine.core.hp / engine.core.max_hp))
            if done:
                break
        limit = engine.cfg.max_steps
        for b in range(TIME_BINS):
            t = (b + 1) * limit / TIME_BINS
            # after the episode ends the Core keeps its final value (0 if destroyed)
            past = [hp for step, hp in core_trace if step <= t]
            core_curves[ep, b] = past[-1] if past else 1.0
        episodes.append({"seed": seed, "win": info["win"], "kills": info["kills"], "length": engine.step_count,
                         "core_hp": info["core_hp_ratio"]})
    env.close()
    return {
        "stage": stage,
        "roles": roles,
        "n_episodes": n_episodes,
        "grid": list(GRID),
        "heatmap": heat.tolist(),
        "confusion": confusion.tolist(),
        "abilities_per_episode": (abilities / n_episodes).tolist(),
        "core_distance": {"mean": [float(np.mean(d)) if d else None for d in dist],
                          "hist": [np.histogram(d, bins=24, range=(0, 720))[0].tolist() if d else [] for d in dist]},
        "core_hp_curve": {"mean": core_curves.mean(0).tolist(), "std": core_curves.std(0).tolist()},
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="run directory")
    parser.add_argument("--script", help="scripted policy name from run_marl_baselines.POLICIES")
    parser.add_argument("--stage", type=int, help="stage for --script")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.run:
        from benchmarks.official.load_policy import load_policy

        policy, env_kwargs = load_policy(args.run)
        label = args.run
    else:
        from benchmarks.runners.run_marl_baselines import POLICIES

        policy, env_kwargs, label = POLICIES[args.script](seed=0), {"stage": args.stage}, args.script
    result = {"source": label, **analyze(policy, env_kwargs, args.episodes)}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    wins = np.mean([e["win"] for e in result["episodes"]])
    print(f"{label}: {args.episodes} test episodes, win {100 * wins:.0f}%, abilities/ep {result['abilities_per_episode']}")


if __name__ == "__main__":
    main()
