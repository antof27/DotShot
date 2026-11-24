"""Scripted Stage 3-4 baselines: random, noop, charge (suicide), and heuristics.

Every learner table must include these rows. They are evaluated through
``benchmarks.eval.marl_eval`` exactly like the trained policies.

Usage:
    python -m benchmarks.runners.run_marl_baselines --config configs/baselines/stage3_baselines.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from benchmarks.eval.evaluate import compute_rliable_stats
from benchmarks.eval.marl_eval import evaluate_marl
from benchmarks.utils.seeding import get_git_info
from dotshot.core.config import ELEMENT_WEAKNESS
from dotshot.core.engine import DotShotEngine
from dotshot.core.entities import AgentState, Enemy
from dotshot.envs.multi_agent import DotShotParallelEnv

NOOP = np.array([1, 1, 0, 0], dtype=np.int64)


def aim_bin(dx: float, dy: float, bins: int) -> int:
    """Aim bin for a direction in screen coordinates (y points down)."""
    ang = float(np.arctan2(-dy, dx)) % (2.0 * np.pi)
    return int(round(ang / (2.0 * np.pi / bins))) % bins


def move_toward(dx: float, dy: float, dead: float = 20.0) -> tuple:
    """MultiDiscrete move components (0, 1, 2) for an 8-way step along (dx, dy)."""
    mx = 2 if dx > dead else (0 if dx < -dead else 1)
    my = 2 if dy > dead else (0 if dy < -dead else 1)
    return mx, my


def lead_point(agent: AgentState, enemy: Enemy) -> tuple:
    """Two fixed-point iterations of the linear intercept point."""
    px, py = enemy.x, enemy.y
    for _ in range(2):
        t = float(np.hypot(px - agent.x, py - agent.y)) / max(1.0, agent.bullet_speed)
        px, py = enemy.x + enemy.vx * t, enemy.y + enemy.vy * t
    return px, py


def fire_action(engine: DotShotEngine, agent: AgentState, enemy: Enemy, mx: int, my: int) -> np.ndarray:
    px, py = lead_point(agent, enemy)
    weapon = ELEMENT_WEAKNESS[enemy.enemy_type] + 1
    return np.array([mx, my, aim_bin(px - agent.x, py - agent.y, engine.cfg.aim_bins), weapon], dtype=np.int64)


class ScriptedPolicy:
    """Base class: per-agent rule on the engine state. Dead agents send noop."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env: DotShotParallelEnv) -> None:
        pass

    def act(self, env: DotShotParallelEnv, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        engine = env.engine
        out = {}
        for name in env.agents:
            agent = engine.agents[env._agent_idx_map[name]]
            out[name] = self.agent_action(engine, agent) if agent.alive else NOOP.copy()
        return out

    def agent_action(self, engine: DotShotEngine, agent: AgentState) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(ScriptedPolicy):
    def agent_action(self, engine, agent):
        return self.rng.integers(0, [3, 3, engine.cfg.aim_bins, 4]).astype(np.int64)


class NoopPolicy(ScriptedPolicy):
    """Stand still and hold fire (in Stage 4, holding fire triggers the class ability)."""

    def agent_action(self, engine, agent):
        return NOOP.copy()


class ChargePolicy(ScriptedPolicy):
    """Walk into the nearest enemy and never fire: the early-termination probe."""

    def agent_action(self, engine, agent):
        enemies = [e for e in engine.enemies if e.alive]
        if not enemies:
            return NOOP.copy()
        e = min(enemies, key=lambda e: (e.x - agent.x) ** 2 + (e.y - agent.y) ** 2)
        mx, my = move_toward(e.x - agent.x, e.y - agent.y, dead=2.0)
        return np.array([mx, my, 0, 0], dtype=np.int64)


def _visible(engine: DotShotEngine, agent: AgentState) -> List[Enemy]:
    return [e for e in engine.enemies if e.alive and engine._is_enemy_visible(agent, e)]


class NearestPolicy(ScriptedPolicy):
    """Shoot the nearest visible enemy with the counter element, lead the target, kite."""

    def agent_action(self, engine, agent):
        visible = _visible(engine, agent)
        if not visible:
            core = engine.core
            mx, my = move_toward(core.x - agent.x, core.y - agent.y, dead=120.0) if core else (1, 1)
            return np.array([mx, my, 0, 0], dtype=np.int64)
        e = min(visible, key=lambda e: (e.x - agent.x) ** 2 + (e.y - agent.y) ** 2)
        dx, dy = e.x - agent.x, e.y - agent.y
        dist = float(np.hypot(dx, dy))
        mx, my = 1, 1
        if dist < 220.0:
            mx, my = move_toward(-dx, -dy)
        elif dist > 460.0:
            mx, my = move_toward(dx, dy)
        return fire_action(engine, agent, e, mx, my)


class CoreGuardPolicy(ScriptedPolicy):
    """Stay near the core and shoot the visible enemy closest to it."""

    def agent_action(self, engine, agent):
        core = engine.core
        mx, my = move_toward(core.x - agent.x, core.y - agent.y, dead=140.0)
        visible = _visible(engine, agent)
        if not visible:
            return np.array([mx, my, 0, 0], dtype=np.int64)
        e = min(visible, key=lambda e: (e.x - core.x) ** 2 + (e.y - core.y) ** 2)
        return fire_action(engine, agent, e, mx, my)


class CalibratedPolicy(ScriptedPolicy):
    """v2.2 reference script: dodge, guard the Core, lead the priority threat, use abilities.

    It reads the engine state directly (privileged), so it is a difficulty yardstick,
    not a fair competitor; it only uses information the v2.2 observations also expose.
    """

    GUARD_RADIUS = 240.0
    DODGE_TICKS = 24.0

    def agent_action(self, engine, agent):
        core = engine.core
        target = engine._priority_threat(agent)
        mx, my = self._move(engine, agent, target)
        if target is None:
            return np.array([mx, my, 0, 0], dtype=np.int64)
        if engine.cfg.stage == 4 and agent.ability_cooldown == 0 and agent.fire_cooldown > 0:
            if self._ability_useful(engine, agent):
                return np.array([mx, my, 0, 0], dtype=np.int64)
        px, py = engine._lead_point(agent, target)
        weapon = ELEMENT_WEAKNESS[target.enemy_type] + 1
        return np.array([mx, my, aim_bin(px - agent.x, py - agent.y, engine.cfg.aim_bins), weapon], dtype=np.int64)

    def _move(self, engine, agent, target):
        # 1. Dodge the most imminent projectile that would hit.
        best = None
        for eb in engine.enemy_bullets:
            if not eb.active:
                continue
            rx, ry = eb.x - agent.x, eb.y - agent.y
            vx, vy = eb.dx * eb.speed, eb.dy * eb.speed
            closing = -(rx * vx + ry * vy)
            if closing <= 0.0:
                continue
            tca = closing / max(1e-6, vx * vx + vy * vy)
            mx_, my_ = rx + vx * tca, ry + vy * tca  # bullet position relative to agent at closest approach
            if tca < self.DODGE_TICKS and np.hypot(mx_, my_) < agent.radius + 14.0 and (best is None or tca < best[0]):
                best = (tca, mx_, my_, vx, vy)
        if best is not None:
            _, mx_, my_, vx, vy = best
            px, py = -vy, vx  # perpendicular to the bullet path, away from the closest-approach point
            if px * mx_ + py * my_ > 0.0:
                px, py = -px, -py
            return move_toward(px, py, dead=0.1)
        # 2. Keep clear of contact range.
        close = [e for e in engine.enemies if e.alive and np.hypot(e.x - agent.x, e.y - agent.y) < e.radius + agent.radius + 70.0]
        if close:
            e = min(close, key=lambda e: (e.x - agent.x) ** 2 + (e.y - agent.y) ** 2)
            return move_toward(agent.x - e.x, agent.y - e.y, dead=0.1)
        # 3. Hold a guard ring around the Core, kiting the target inside it.
        core = engine.core
        if core is not None and np.hypot(core.x - agent.x, core.y - agent.y) > self.GUARD_RADIUS:
            return move_toward(core.x - agent.x, core.y - agent.y)
        if target is not None:
            dx, dy = target.x - agent.x, target.y - agent.y
            dist = float(np.hypot(dx, dy))
            if dist < 200.0:
                return move_toward(-dx, -dy)
            if dist > 520.0 and (core is None or np.hypot(target.x - core.x, target.y - core.y) < 2 * self.GUARD_RADIUS):
                return move_toward(dx, dy)
        return 1, 1

    @staticmethod
    def _ability_useful(engine, agent) -> bool:
        from dotshot.core.config import ROLE_SUPPORT, ROLE_VANGUARD

        near = [e for e in engine.enemies if e.alive and np.hypot(e.x - agent.x, e.y - agent.y) < 380.0]
        if agent.role_id == ROLE_VANGUARD:
            return len(near) >= 2 or (engine.core is not None and engine.core.under_attack_timer > 0 and bool(near))
        if agent.role_id == ROLE_SUPPORT:
            core = engine.core
            hurt = core is not None and core.hp < 0.9 * core.max_hp
            return hurt or any(a.alive and a.hp < 0.8 * a.max_hp for a in engine.agents) or bool(near)
        return False


POLICIES = {
    "random": RandomPolicy,
    "noop": NoopPolicy,
    "charge": ChargePolicy,
    "heuristic_nearest": NearestPolicy,
    "heuristic_core_guard": CoreGuardPolicy,
    "heuristic_calibrated": CalibratedPolicy,
}


def run_baselines(cfg: dict, out_dir: Optional[str] = None) -> Dict[str, dict]:
    stage = int(cfg["stage"])
    env_kwargs = {
        "stage": stage,
        "partial_obs": cfg.get("partial_obs"),
        "reward_alpha": cfg.get("reward_alpha"),
        "max_steps": cfg.get("max_steps"),
        "reward_mode": cfg.get("reward_mode"),
    }
    split = str(cfg.get("split", "test"))
    n_episodes = int(cfg.get("n_episodes", 32))
    out = Path(out_dir or cfg.get("out_dir", f"runs/baselines/stage{stage}"))
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}
    for name in cfg.get("policies", list(POLICIES)):
        policy = POLICIES[name](seed=int(cfg.get("seed", 0)))
        metrics, episodes = evaluate_marl(env_kwargs, policy, n_episodes, split=split)
        results[name] = {
            "metrics": metrics,
            "return_stats": compute_rliable_stats([e["return"] for e in episodes]),
            "win_stats": compute_rliable_stats([e["win"] for e in episodes]),
        }
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "metrics": metrics, "episodes": episodes}, f, indent=2)
        r = results[name]["return_stats"]
        print(
            f"[stage {stage}] {name:22s} return {r['mean']:7.2f} [{r['mean_ci_low']:.2f}, {r['mean_ci_high']:.2f}] "
            f"win {metrics['eval/win_rate']*100:5.1f}% kills {metrics['eval/kills_mean']:5.1f} "
            f"core {metrics['eval/core_hp_ratio']*100:5.1f}% len {metrics['eval/episode_length']:6.0f}"
        )

    summary = {"config": cfg, "git": get_git_info("."), "results": results}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scripted DotShot Stage 3-4 baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_baselines(cfg, args.out_dir)


if __name__ == "__main__":
    main()
