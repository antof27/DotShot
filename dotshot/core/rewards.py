"""Decomposed individual and team reward engine for DotShot Game v2."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AgentStepEvents:
    """Raw per-step combat events accumulated for one agent."""
    hits_correct: int = 0
    hits_wrong: int = 0
    precision_hits: int = 0
    kills: int = 0
    boss_kills: int = 0
    shield_breaks: int = 0
    combo_kills: int = 0
    missed_bullets: int = 0
    well_aimed_misses: int = 0
    ally_hits: int = 0
    ally_kills: int = 0
    damage_taken: float = 0.0
    shield_absorbed: float = 0.0
    heal_utility: float = 0.0
    mark_utility: int = 0
    aim_alignment: float = 0.0
    trigger_discipline: float = 0.0
    proximity_danger: float = 0.0
    wall_bump: float = 0.0
    died: bool = False
    alive: bool = True


@dataclass
class TeamStepEvents:
    """Shared team-level tactical events per step."""
    enemies_killed: int = 0
    bosses_killed: int = 0
    shields_broken: int = 0
    combo_kills: int = 0
    core_damage_taken: float = 0.0
    core_healed: float = 0.0
    core_destroyed: bool = False
    any_agent_died: bool = False
    wave_cleared: bool = False
    core_alive: bool = True


def compute_decomposed_rewards(
    agent_events: Dict[int, AgentStepEvents],
    team_events: TeamStepEvents,
    alpha: float,
    num_agents: int,
    episode_misses: Dict[int, int],
) -> tuple[Dict[int, float], Dict[str, float]]:
    """
    Compute per-agent reward using explicit individual/team mixing:
        r_i^t = (1 - alpha) * r_{i, indiv}^t + alpha * r_team^t

    Returns (per_agent_rewards, aggregate_breakdown_dict).
    """
    # 1. Shared team signal
    r_team_raw = (
        2.5 * team_events.enemies_killed
        + 8.0 * team_events.bosses_killed
        + 3.5 * team_events.shields_broken
        + 4.0 * team_events.combo_kills
        + 6.0 * (1.0 if team_events.wave_cleared else 0.0)
        + 0.04 * team_events.core_healed
        - 0.08 * team_events.core_damage_taken
        - (12.0 if team_events.core_destroyed else 0.0)
        - (3.0 if team_events.any_agent_died else 0.0)
        + (0.02 if team_events.core_alive else 0.0)
    )
    r_team = r_team_raw / 10.0

    rewards: Dict[int, float] = {}
    indiv_sum = 0.0

    for idx, ev in agent_events.items():
        if not ev.alive and not ev.died:
            rewards[idx] = 0.0
            continue

        miss_mult = min(3.0, 1.0 + 0.1 * episode_misses.get(idx, 0))
        r_indiv_raw = (
            1.2 * ev.hits_correct
            + 0.6 * ev.precision_hits
            + 3.0 * ev.kills
            + 8.0 * ev.boss_kills
            + 3.0 * ev.shield_breaks
            + 3.5 * ev.combo_kills
            + 0.12 * ev.aim_alignment
            + 0.15 * ev.trigger_discipline
            + 0.05 * ev.shield_absorbed
            + 0.05 * ev.heal_utility
            + 0.8 * ev.mark_utility
            + (0.015 if ev.alive else 0.0)
            - 0.25 * ev.hits_wrong
            - 0.22 * miss_mult * ev.missed_bullets
            - 0.02 * ev.well_aimed_misses
            - 2.0 * ev.ally_hits
            - 5.0 * ev.ally_kills
            - 0.20 * ev.damage_taken
            - 0.15 * ev.proximity_danger
            - 0.08 * ev.wall_bump
            - (5.0 if ev.died else 0.0)
        )
        r_indiv = r_indiv_raw / 10.0
        indiv_sum += r_indiv

        if num_agents == 1:
            rewards[idx] = float(r_indiv)
        else:
            rewards[idx] = float((1.0 - alpha) * r_indiv + alpha * r_team)

    breakdown = {
        "r_team": float(r_team),
        "r_indiv_mean": float(indiv_sum / max(1, len(agent_events))),
    }
    return rewards, breakdown
