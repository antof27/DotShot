def calculate_reward(env, metrics):
    """
    Balanced reward function for Element Shooter game.

    Design principles:
    - Positive signals (hits, kills, aiming, survival) must be reachable by a learning agent
    - Penalties discourage bad behavior but don't drown out positive signals
    - No reward clamping — let the value function learn the true distribution
    """
    # 1. Penalties (negative values)
    # Contact damage taken (increased to strongly discourage melee range)
    contact_damage_penalty = -2.0 * metrics.get("damage_taken", 0.0)
    
    # Projectile damage taken (from enemy bullets)
    bullet_damage_penalty = -1.5 * metrics.get("eb_damage_taken", 0.0)
    
    # Death penalty
    death_penalty = -10.0 if metrics.get("any_dead", False) else 0.0
    
    # Edge/corner camping penalty
    corner_camping_penalty = -0.5 * metrics.get("edge_penalty", 0.0)
    
    # Proximity danger penalty — continuous penalty for being within 150px of hostile enemies.
    # Encourages kiting and maintaining distance before contact damage even occurs.
    proximity_danger_penalty = -1.5 * metrics.get("proximity_danger", 0.0)
    
    # Friendly fire penalties (painful but not catastrophic — agent can recover and learn)
    ally_hit_penalty = -8.0 * metrics.get("ally_hit", 0.0)
    ally_killed_penalty = -20.0 * metrics.get("ally_killed", 0.0)
    
    # Soft ally alignment penalties to prevent friendly fire (Option C)
    ally_aim_warning_penalty = -0.5 * metrics.get("ally_aim_penalty", 0.0)
    ally_shoot_warning_penalty = -0.5 * metrics.get("ally_shoot_penalty", 0.0)
    
    # Escalating bullet miss penalty — each new miss costs more as cumulative waste grows.
    # The multiplier grows by 20% per prior miss: 1st miss = -1.5, 5th = -3.0, 10th = -4.5, etc.
    episode_misses = metrics.get("episode_missed_bullets", 0.0)
    waste_multiplier = 1.0 + 0.2 * episode_misses
    bullet_miss_penalty = -1.5 * waste_multiplier * metrics.get("missed_bullets_count", 0.0)
    
    # Well-aimed bullet miss penalty (near-zero to avoid discouraging aimed shots)
    well_aimed_miss_penalty = -0.05 * metrics.get("well_aimed_misses_count", 0.0)
    
    # No bullet_fired_penalty — the miss penalty already handles spamming.
    # Charging per shot punishes correct hits too.
    
    # Wrong element matching penalty
    incorrect_hit_penalty = -1.5 * metrics.get("enemy_hit_incorrect", 0.0)
    
    # 2. Rewards (positive values)
    # Correct weapon hits (boosted to make hits clearly rewarding)
    correct_hits_reward = 8.0 * metrics.get("enemy_hit_correct", 0.0)
    
    # Hostile kills (boosted — the primary goal)
    hostile_kills_reward = 25.0 * metrics.get("enemy_killed", 0.0)
    
    # Aiming alignment reward (already boosted in game_env.py)
    alignment_reward = metrics.get("alignment_reward", 0.0)
    
    # Trigger discipline reward (bonus for NOT shooting when poorly aimed)
    trigger_discipline_reward = 1.0 * metrics.get("trigger_discipline_reward", 0.0)
    
    # Precision hit bonus (extra reward for well-aimed bullets that actually connect)
    precision_hit_bonus = 4.0 * metrics.get("precision_hits", 0.0)
    
    # Survival bonus (doubled for clearer per-step positive signal)
    survival_bonus = metrics.get("survival_bonus", 0.1)
    
    # Calculate the total reward
    total_reward = (
        contact_damage_penalty +
        bullet_damage_penalty +
        death_penalty +
        corner_camping_penalty +
        proximity_danger_penalty +
        ally_hit_penalty +
        ally_killed_penalty +
        ally_aim_warning_penalty +
        ally_shoot_warning_penalty +
        bullet_miss_penalty +
        well_aimed_miss_penalty +
        incorrect_hit_penalty +
        correct_hits_reward +
        hostile_kills_reward +
        alignment_reward +
        trigger_discipline_reward +
        precision_hit_bonus +
        survival_bonus
    )
    
    # Scale down reward to reduce value loss variance, stabilizing training
    scaled_reward = total_reward / 10.0
    
    # No clamping — let the value function learn the true reward distribution
    return scaled_reward