import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, obs_dim=55, action_dim=8, latent_dim=64):
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
    def __init__(self, env, eta=0.1, beta=0.2, lr=1e-4):
        """
        Gymnasium Environment Wrapper that calculates and injects intrinsic curiosity rewards.
        eta: Intrinsic reward scaling factor
        beta: Weighing factor between forward and inverse loss (0.2 means 20% forward, 80% inverse)
        """
        super(CuriosityWrapper, self).__init__(env)
        
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.latent_dim = 64
        
        self.eta = eta
        self.beta = beta
        
        # Initialize ICM PyTorch networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.icm = IntrinsicCuriosityModule(self.obs_dim, self.action_dim, self.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.icm.parameters(), lr=lr)
        
        self.mse_loss = nn.MSELoss()
        
        self.last_obs = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        return obs, info
        
    def step(self, action):
        # 1. Step the physical environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Calculate Intrinsic Reward
        if self.last_obs is not None:
            # Convert to PyTorch tensors
            s_t = torch.FloatTensor(self.last_obs).unsqueeze(0).to(self.device)
            s_t1 = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            a_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            # Forward pass through ICM
            phi_next_state, pred_phi_next_state, pred_action = self.icm(s_t, s_t1, a_t)
            
            # Intrinsic reward = mean squared error between actual and predicted feature representation
            forward_loss = self.mse_loss(pred_phi_next_state, phi_next_state.detach())
            intrinsic_reward = (self.eta / 2.0) * forward_loss.item()
            
            # Inverse loss: how well did we predict the action that was taken
            inverse_loss = self.mse_loss(pred_action, a_t)
            
            # Total ICM training loss
            loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
            
            # Update ICM weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Add intrinsic curiosity to the environment reward
            reward = float(reward) + intrinsic_reward
            
            # Log curiosity statistics in info dict for debugging
            info["intrinsic_reward"] = intrinsic_reward
            info["icm_loss"] = loss.item()
            info["forward_loss"] = forward_loss.item()
            info["inverse_loss"] = inverse_loss.item()
            
        self.last_obs = next_obs.copy()
        return next_obs, reward, terminated, truncated, info
