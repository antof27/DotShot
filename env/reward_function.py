def calculate_reward(env, metrics):
    # Calculate contact damage penalty
    contact_penalty = -1.1 * (metrics['damage_taken'] / env.max_health) if metrics['agent_healths'][0] > 0 else 0
    
    # Calculate bullet damage penalty
    bullet_penalty = -1.5 * (metrics['enemy_bullets_count'] / env.max_health) if metrics['agent_healths'][0] > 0 else 0

    # Check for death and apply penalty
    death_penalty = -10.0 if any(agent_health == 0 for agent_health in metrics['agent_healths']) else 0
    
    # Calculate edge penalty (corner camping)
    edge_penalty = -0.08 * abs(metrics['edge_penalty'])
    
    # Check for friendly allies hit and apply penalty
    ally_hit_penalty = -15.0 if metrics['ally_hit'] > 0 else 0
    ally_killed_penalty = -25.0 if metrics['ally_killed'] > 0 else 0
    
    # Calculate correct hits reward
    correct_hits_reward = max(3.0, min(metrics['enemy_hit_correct'], (metrics['alignment_reward'] * 2) / metrics['difficulty']))
    
    # Check for aiming alignment and apply reward
    if metrics['alignment_reward'] > 0:
        alignment_bonus = max(1.8, min(metrics['alignment_reward'], (metrics['alignment_reward'] - 0.7) * 2.6))
    else:
        alignment_bonus = 0
    
    # Calculate bullet misses penalty
    missed_bullets_penalty = -2.0 * metrics['missed_bullets_count']
    
    # Reward survival and calculate total reward
    total_reward = -contact_penalty - bullet_penalty + correct_hits_reward + alignment_bonus - missed_bullets_penalty + (metrics['steps_survived'] / 3600) * metrics['survival_bonus']
    if any(agent_health == 0 for agent_health in metrics['agent_healths']):
        total_reward -= death_penalty
    total_reward += edge_penalty
    
    return total_reward