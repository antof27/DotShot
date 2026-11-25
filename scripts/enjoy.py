"""Play DotShot by hand, or replay a trained run and optionally record it.

Examples:
    python scripts/enjoy.py --stage 3 --manual
    python scripts/enjoy.py --run runs/final/stage3/s3_harl_mappo_bc/seed_0
    python scripts/enjoy.py --run runs/final/stage4/s4_mat_bc/seed_0 --record docs/assets/videos/stage4_mat.mp4

Replays render every simulator tick. The environment runs with ``action_repeat: 1``
and the script itself holds each policy action for the run's decision interval,
which reproduces the training interface exactly while giving smooth video.
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from dotshot.core.config import get_stage_config
from dotshot.envs.multi_agent import DotShotParallelEnv
from dotshot.envs.single_agent import DotShotSingleEnv


def _manual_action(engine, weapon: int) -> np.ndarray:
    import pygame

    pygame.event.pump()
    keys = pygame.key.get_pressed()
    mx = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
    my = (keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP])
    mouse_x, mouse_y = pygame.mouse.get_pos()
    agent = engine.agents[0]
    ang = float(np.arctan2(-(mouse_y - agent.y), mouse_x - agent.x)) % (2.0 * np.pi)
    aim = int(round(ang / (2.0 * np.pi / engine.cfg.aim_bins))) % engine.cfg.aim_bins
    shoot = pygame.mouse.get_pressed()[0] or keys[pygame.K_SPACE]
    return np.array([mx + 1, my + 1, aim, weapon + 1 if shoot else 0], dtype=np.int64)


def run_manual(stage: int) -> None:
    import pygame

    print("Move: WASD | Aim: mouse | Element: 1 Water, 2 Grass, 3 Fire | Fire: click or Space | Overlay: Tab")
    tick = {"action_repeat": 1}  # humans act every tick
    multi = stage >= 3
    env = (DotShotParallelEnv if multi else DotShotSingleEnv)(stage=stage, render_mode="human", env_overrides=tick)
    env.reset()
    weapon = 0
    try:
        while True:
            keys = pygame.key.get_pressed()
            for k, w in ((pygame.K_1, 0), (pygame.K_2, 1), (pygame.K_3, 2)):
                if keys[k]:
                    weapon = w
            action = _manual_action(env.engine, weapon)
            if multi:
                actions = {a: np.array([1, 1, 0, 0], dtype=np.int64) for a in env.possible_agents}
                actions["agent_0"] = action
                _, _, terms, truncs, infos = env.step(actions)
                done, info = any(terms.values()) or any(truncs.values()), next(iter(infos.values()))
            else:
                _, _, term, trunc, info = env.step(action)
                done = term or trunc
            if env._renderer and env._renderer.quit_requested:
                break
            if done:
                print(f"Episode over | kills {info['kills']:.0f} | win {info['win']:.0f}")
                env.reset()
    finally:
        env.close()


def run_replay(run_dir: str, which: str, seed: int, record: str | None, frame_skip: int, fps: int,
               snapshots: str | None = None, snap_ticks: tuple = ()) -> None:
    from benchmarks.eval.marl_eval import episode_seeds
    from benchmarks.official.load_policy import load_policy

    offscreen = bool(record or snapshots)
    if offscreen:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    policy, env_kwargs = load_policy(run_dir, which)
    stage = int(env_kwargs["stage"])
    overrides = dict(env_kwargs.get("env_overrides") or {})
    repeat = int(overrides.get("action_repeat", get_stage_config(stage).action_repeat))
    overrides["action_repeat"] = 1
    kwargs = {k: v for k, v in env_kwargs.items() if k not in ("stage", "env_overrides") and v is not None}
    multi = stage >= 3
    env = (DotShotParallelEnv if multi else DotShotSingleEnv)(
        stage=stage, render_mode="rgb_array" if offscreen else "human", env_overrides=overrides, **kwargs
    )
    ep_seed = episode_seeds("test", seed + 1)[seed]  # the seed-th held-out test episode
    obs, _ = env.reset(seed=ep_seed)
    policy.reset(env)
    frames, action, t = [], None, 0
    pending, seen = {}, set()  # callout text -> tick to save; texts already saved
    try:
        while True:
            if t % repeat == 0:
                action = policy.act(env, obs)
            if multi:
                obs, _, terms, truncs, infos = env.step(action)
                done, info = any(terms.values()) or any(truncs.values()), infos[env.possible_agents[0]]
            else:
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            if snapshots:
                for ev in env.engine.visual_events:
                    text = ev.get("text")
                    if ev.get("type") == "callout" and text and text not in seen and text not in pending:
                        pending[text] = t + 4  # a few ticks later the effect is fully drawn
            frame = env.render() if (record and t % frame_skip == 0) or snapshots else None
            if record and t % frame_skip == 0:
                frames.append(frame)
            elif not offscreen and env._renderer and env._renderer.quit_requested:
                break
            if snapshots:
                due = [k for k, v in pending.items() if v == t] + ([f"tick{t}"] if t in snap_ticks else [])
                for name in due:
                    _save_png(frame, snapshots, name)
                    seen.add(name)
                    pending.pop(name, None)
            t += 1
            if done:
                break
    finally:
        env.close()
    core = f" | Core HP {100 * info['core_hp_ratio']:.0f}%" if multi else ""
    print(f"test episode {seed} (seed {ep_seed}): {t} ticks | kills {info['kills']:.0f} | win {info['win']:.0f}{core}")
    if record and frames:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(record)), exist_ok=True)
        imageio.mimsave(record, frames, fps=fps)
        print(f"saved {len(frames)} frames to {record}")


def _save_png(frame, folder: str, name: str) -> None:
    import imageio.v2 as imageio

    os.makedirs(folder, exist_ok=True)
    slug = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
    imageio.imwrite(os.path.join(folder, f"{slug}.png"), frame)
    print(f"snapshot {name} -> {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="stage for --manual")
    parser.add_argument("--manual", action="store_true", help="play with keyboard and mouse")
    parser.add_argument("--run", help="run directory with config.yaml and checkpoints/")
    parser.add_argument("--which", default="final", choices=["final", "best"])
    parser.add_argument("--episode", type=int, default=0, help="index of the held-out test episode to replay")
    parser.add_argument("--record", help="write an MP4 or GIF instead of opening a window")
    parser.add_argument("--frame-skip", type=int, default=2, help="record every n-th tick")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--snapshots", help="folder for PNGs of the first SHIELD BREAK, AEGIS, PULSE, COMBO KILL")
    parser.add_argument("--snap-ticks", default="", help="extra ticks to capture, e.g. 400,1500")
    args = parser.parse_args()
    if args.manual and args.stage:
        run_manual(args.stage)
    elif args.run:
        ticks = tuple(int(x) for x in args.snap_ticks.split(",") if x)
        run_replay(args.run, args.which, args.episode, args.record, args.frame_skip, args.fps, args.snapshots, ticks)
    else:
        parser.error("use --manual --stage N, or --run RUN_DIR")


if __name__ == "__main__":
    main()
