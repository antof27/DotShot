import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.callbacks import BaseCallback

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, obs_dim=60, action_dim=6, latent_dim=64):
        super(IntrinsicCuriosityModule, self).__init__()
        
        # 1. Feature Encoder: φ(s) -> maps high-dim state to low-dim features
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # 2. Inverse Dynamics Model: g(φ(s_t), φ(s_t+1)) -> predicts action a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Actions are bounded in [-1, 1]
        )
        
        # 3. Forward Dynamics Model: f(φ(s_t), a_t) -> predicts φ(s_t+1)
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, state, next_state, action):
        # Encode states
        phi_state = self.encoder(state)
        phi_next_state = self.encoder(next_state)
        
        # Inverse model prediction: g(φ(s), φ(s_next)) -> a_pred
        inverse_input = torch.cat([phi_state, phi_next_state], dim=-1)
        pred_action = self.inverse_model(inverse_input)
        
        # Forward model prediction: f(φ(s), a) -> φ(s_next)_pred
        forward_input = torch.cat([phi_state, action], dim=-1)
        pred_phi_next_state = self.forward_model(forward_input)
        
        return phi_next_state, pred_phi_next_state, pred_action


class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env, eta=0.1, beta=0.2, lr=1e-4, R_min=None, R_max=None):
        """
        Gymnasium Environment Wrapper that implements Dynamic Potential-Based Reward Shaping (PBRS)
        combining aim alignment and kiting distance.
        Note: The neural-network based Intrinsic Curiosity Module (ICM) has been decoupled
        and moved to CuriosityCallback to ensure correct vectorized training.
        """
        super(CuriosityWrapper, self).__init__(env)
        
        self.eta = eta
        self.beta = beta
        self.R_min = R_min
        self.R_max = R_max
        
        # Initialize running rewards for auto-scaling
        self.running_min_rew = R_min
        self.running_max_rew = R_max
        
        # PBRS parameters
        self.num_agents = getattr(self.env.unwrapped, "num_agents", 1)
        self.prev_potential = [0.0] * self.num_agents
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = [0.0] * self.num_agents
        
        # Dynamic scaling: auto-track bounds if R_min or R_max is None
        unwrapped = self.env.unwrapped
        mean_rew = getattr(unwrapped, "mean_episode_reward", 0.0)
        
        if self.R_min is None:
            if self.running_min_rew is None:
                self.running_min_rew = mean_rew
            else:
                self.running_min_rew = min(self.running_min_rew, mean_rew)
        if self.R_max is None:
            if self.running_max_rew is None:
                self.running_max_rew = mean_rew
            else:
                self.running_max_rew = max(self.running_max_rew, mean_rew)
                
        return obs, info
        
    def step(self, action):
        # 1. Step the physical environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Calculate Dynamic Potential-Based Reward Shaping (PBRS)
        unwrapped = self.env.unwrapped
        
        # Dynamic scaling factor based on rolling average episode reward
        mean_rew = getattr(unwrapped, "mean_episode_reward", 0.0)
        
        # Use running bounds if fixed bounds are not provided
        r_min = self.running_min_rew if self.R_min is None else self.R_min
        r_max = self.running_max_rew if self.R_max is None else self.R_max
        
        if r_min is None or r_max is None or abs(r_max - r_min) < 1e-5:
            k = 0.0
        else:
            k = np.clip((mean_rew - r_min) / (r_max - r_min), 0.0, 1.0)
        shaping_scale = 0.5 * (1.0 - 0.8 * k)  # Scales from 0.5 (struggling) to 0.1 (mastered)
        
        pbrs_reward = 0.0
        for idx in range(self.num_agents):
            # Parse aiming direction angle from action
            agent_action = action[idx*4 : (idx+1)*4]
            aim_angle = float(agent_action[2]) * (2.0 * np.pi / 64.0)
            aim_x = np.cos(aim_angle)
            aim_y = -np.sin(aim_angle)
            
            ax = unwrapped.agent_x[idx]
            ay = unwrapped.agent_y[idx]
            
            # Find best alignment and distance to nearest active hostile enemy
            best_alignment = -1.0
            closest_distance = float('inf')
            active_enemies = [e for e in unwrapped.enemies if e.alive and not getattr(e, 'is_ally', False)]
            
            # Health potential (ranges from 0.0 to 1.0)
            health_potential = unwrapped.agent_health[idx] / unwrapped.max_health
            
            # Center potential (ranges from 0.0 to 1.0, higher near the center)
            center_x = unwrapped.width / 2.0
            center_y = unwrapped.height / 2.0
            dist_from_center = np.sqrt((ax - center_x)**2 + (ay - center_y)**2)
            max_dist_center = np.sqrt(center_x**2 + center_y**2)
            center_potential = 1.0 - (dist_from_center / max_dist_center)
            
            if active_enemies:
                for enemy in active_enemies:
                    ex = enemy.x - ax
                    ey = enemy.y - ay
                    edist = np.sqrt(ex**2 + ey**2)
                    if edist > 0:
                        alignment = (aim_x * ex + aim_y * ey) / edist
                        best_alignment = max(best_alignment, alignment)
                    if edist < closest_distance:
                        closest_distance = edist
                
                # Normalize distance (max possible distance on 800x800 map is sqrt(800^2 + 800^2) = 1131.37)
                norm_dist = min(1.0, closest_distance / 1131.37)
                
                align_potential = max(0.0, best_alignment)
                dist_potential = norm_dist
                
                # Combined Potential: 40% Aiming, 30% Kiting distance, 20% Agent Health, 10% Center proximity
                current_potential = (
                    0.4 * align_potential +
                    0.3 * dist_potential +
                    0.2 * health_potential +
                    0.1 * center_potential
                )
            else:
                # No active enemies: focus on survival and staying centered
                current_potential = 0.6 * health_potential + 0.4 * center_potential
                
            # gamma * Phi(s') - Phi(s) where gamma = 0.99
            pbrs_val = 0.99 * current_potential - self.prev_potential[idx]
            self.prev_potential[idx] = current_potential
            
            # Accumulate scaled shaped reward
            pbrs_reward += pbrs_val * shaping_scale
            
        # Log PBRS values in info
        info["pbrs_reward"] = pbrs_reward
        info["pbrs_shaping_scale"] = shaping_scale
        
        # Add shaped reward to environment reward (Curiosity is added in Callback)
        reward = float(reward) + pbrs_reward
        
        return next_obs, reward, terminated, truncated, info


class CuriosityCallback(BaseCallback):
    def __init__(self, obs_dim, action_dim, latent_dim=64, eta=0.1, beta=0.2, lr=1e-4, device="auto", verbose=0):
        super(CuriosityCallback, self).__init__(verbose)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.eta = eta
        self.beta = beta
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.icm = IntrinsicCuriosityModule(self.obs_dim, self.action_dim, self.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.icm.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
        # Buffer to store trajectories during rollout
        self.rollout_data = []

    def _on_step(self) -> bool:
        # Extract variables from SB3 rollout collection context
        obs = self.model._last_obs
        new_obs = self.locals["new_obs"]
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        
        # Save to buffer for training at the end of rollout
        self.rollout_data.append((obs.copy(), actions.copy(), new_obs.copy()))
        
        # Compute forward pass to calculate intrinsic rewards
        with torch.no_grad():
            s_t = torch.FloatTensor(obs).to(self.device)
            s_t1 = torch.FloatTensor(new_obs).to(self.device)
            
            # Normalize actions if discrete/MultiDiscrete
            action_space = self.training_env.action_space
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                n_vec = action_space.nvec
                action_np = np.array(actions, dtype=np.float32)
                normalized_action = 2.0 * action_np / (n_vec - 1) - 1.0
            elif isinstance(action_space, gym.spaces.Discrete):
                normalized_action = 2.0 * actions / (action_space.n - 1) - 1.0
            else:
                normalized_action = actions
                
            a_t = torch.FloatTensor(normalized_action).to(self.device)
            
            phi_next_state, pred_phi_next_state, _ = self.icm(s_t, s_t1, a_t)
            
            # Compute MSE forward loss per environment
            forward_err = torch.mean((pred_phi_next_state - phi_next_state) ** 2, dim=-1)
            intrinsic_rewards = (self.eta / 2.0) * forward_err.cpu().numpy()
            
        # Update rewards in-place in self.locals
        for i in range(len(rewards)):
            rewards[i] += intrinsic_rewards[i]
            
            # Log curiosity statistics in the info dict
            infos[i]["intrinsic_reward"] = intrinsic_rewards[i]
            
        return True

    def _on_rollout_end(self) -> None:
        if not self.rollout_data:
            return
            
        obs_list, actions_list, next_obs_list = zip(*self.rollout_data)
        
        # Concatenate arrays along batch axis
        obs_all = np.concatenate(obs_list, axis=0)
        actions_all = np.concatenate(actions_list, axis=0)
        next_obs_all = np.concatenate(next_obs_list, axis=0)
        
        dataset_size = obs_all.shape[0]
        batch_size = 256
        epochs = 3
        
        indices = np.arange(dataset_size)
        
        self.icm.train()
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                
                s_t = torch.FloatTensor(obs_all[batch_indices]).to(self.device)
                s_t1 = torch.FloatTensor(next_obs_all[batch_indices]).to(self.device)
                actions_batch = actions_all[batch_indices]
                
                # Normalize actions
                action_space = self.training_env.action_space
                if isinstance(action_space, gym.spaces.MultiDiscrete):
                    n_vec = action_space.nvec
                    action_np = np.array(actions_batch, dtype=np.float32)
                    normalized_action = 2.0 * action_np / (n_vec - 1) - 1.0
                elif isinstance(action_space, gym.spaces.Discrete):
                    normalized_action = 2.0 * actions_batch / (action_space.n - 1) - 1.0
                else:
                    normalized_action = actions_batch
                    
                a_t = torch.FloatTensor(normalized_action).to(self.device)
                
                # Forward + Inverse Loss
                phi_next_state, pred_phi_next_state, pred_action = self.icm(s_t, s_t1, a_t)
                
                forward_loss = self.mse_loss(pred_phi_next_state, phi_next_state.detach())
                inverse_loss = self.mse_loss(pred_action, a_t)
                
                loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # Clear buffer for the next rollout
        self.rollout_data = []

