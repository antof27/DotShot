import torch
import torch.nn as nn
import torch.nn.functional as F

class PMATAgentEncoder(nn.Module):
    """
    Transformer Encoder that processes individual agent observations into representation tokens.
    """
    def __init__(self, obs_dim, hidden_dim, num_heads=4, num_layers=2):
        super(PMATAgentEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs_n):
        """
        obs_n: Shape [batch_size, num_agents, obs_dim]
        Returns: Agent representation tokens of shape [batch_size, num_agents, hidden_dim]
        """
        x = self.feature_extractor(obs_n)  # [batch_size, num_agents, hidden_dim]
        tokens = self.transformer(x)       # [batch_size, num_agents, hidden_dim]
        return tokens


class FixedPrioritizer(nn.Module):
    """
    Fixed/Deterministic agent ordering (MAT).
    """
    def __init__(self):
        super(FixedPrioritizer, self).__init__()

    def forward(self, agent_tokens, deterministic=False):
        batch_size, num_agents, _ = agent_tokens.shape
        device = agent_tokens.device
        
        # Permutation is always [0, 1, 2, ..., num_agents-1]
        permutation = torch.arange(num_agents, device=device).unsqueeze(0).expand(batch_size, -1)
        log_prob = torch.zeros(batch_size, device=device)
        return permutation, log_prob


class PlackettLucePrioritizer(nn.Module):
    """
    Predicts priority logits for each agent and samples execution order 
    using the Plackett-Luce ranking model.
    """
    def __init__(self, hidden_dim):
        super(PlackettLucePrioritizer, self).__init__()
        self.priority_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_tokens, deterministic=False):
        """
        agent_tokens: [batch_size, num_agents, hidden_dim]
        Returns:
          permutation: [batch_size, num_agents] (selected execution order indices)
          log_prob: [batch_size] (log probability of the sampled permutation)
        """
        batch_size, num_agents, _ = agent_tokens.shape
        device = agent_tokens.device
        
        # Predict logits of priority for each agent
        priority_logits = self.priority_network(agent_tokens).squeeze(-1)  # [batch_size, num_agents]
        
        if deterministic:
            # Deterministically sort from highest priority to lowest
            _, permutation = torch.sort(priority_logits, dim=-1, descending=True)
            log_prob = torch.zeros(batch_size, device=device)
            return permutation, log_prob

        permutation = []
        log_prob = torch.zeros(batch_size, device=device)
        masked_logits = priority_logits.clone()
        
        # Plackett-Luce sequential sampling
        for _ in range(num_agents):
            probs = torch.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_agent = dist.sample()  # [batch_size]
            
            permutation.append(next_agent)
            log_prob += dist.log_prob(next_agent)
            
            # Mask out the selected agent so it can't be selected again
            mask = torch.zeros_like(masked_logits)
            mask.scatter_(-1, next_agent.unsqueeze(-1), 1.0)
            masked_logits = masked_logits.masked_fill(mask.bool(), -1e9)
            
        permutation = torch.stack(permutation, dim=1)  # [batch_size, num_agents]
        return permutation, log_prob


class PMATDecoder(nn.Module):
    """
    Transformer Decoder that sequentially predicts action distributions for each agent 
    conditioned on previously selected agents and actions.
    """
    def __init__(self, action_dim, hidden_dim, num_heads=4, num_layers=2):
        super(PMATDecoder, self).__init__()
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, target_agent_tokens, prev_action_embeddings, causal_mask=None):
        """
        target_agent_tokens: [batch_size, num_agents, hidden_dim] (Agent tokens in decision order)
        prev_action_embeddings: [batch_size, num_agents, hidden_dim] (Embedded action history)
        causal_mask: Custom upper triangular causal mask to prevent looking into the future
        """
        # Memory is target agent features (what the agent sees)
        # Query is action history (what has already happened)
        decoded = self.transformer(
            tgt=prev_action_embeddings, 
            memory=target_agent_tokens,
            tgt_mask=causal_mask
        )
        action_logits = self.action_head(decoded)  # [batch_size, num_agents, action_dim]
        return action_logits


class CentralizedCritic(nn.Module):
    """
    Centralized value network that processes global states or joint observations 
    to output centralized state-value estimates.
    """
    def __init__(self, global_state_dim, hidden_dim):
        super(CentralizedCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)


class PMATPolicy(nn.Module):
    """
    Unified Prioritized Multi-Agent Transformer (PMAT) Policy architecture.
    """
    def __init__(self, obs_dim, action_dim, global_state_dim, hidden_dim=128):
        super(PMATPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = PMATAgentEncoder(obs_dim, hidden_dim)
        self.prioritizer = PlackettLucePrioritizer(hidden_dim)
        self.decoder = PMATDecoder(action_dim, hidden_dim)
        self.critic = CentralizedCritic(global_state_dim, hidden_dim)
        
        # Start token for the first agent in the autoregressive sequence
        self.start_action_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def evaluate_actions(self, obs_n, actions_n, global_state):
        """
        Training Forward Pass: Evaluate joint actions under the current policy state.
        Computes exact entropy, joint action log probability, and centralized value estimates.
        """
        batch_size, num_agents, _ = obs_n.shape
        device = obs_n.device
        
        # 1. Encode agent observations
        agent_tokens = self.encoder(obs_n)  # [batch_size, num_agents, hidden_dim]
        
        # 2. Get dynamic execution priorities and sample permutation order
        permutation, perm_log_prob = self.prioritizer(agent_tokens)
        
        # 3. Align agent tokens and actions based on sampled execution permutation
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, num_agents)
        target_agent_tokens = agent_tokens[batch_indices, permutation]  # [batch_size, num_agents, hidden_dim]
        ordered_actions = actions_n[batch_indices, permutation]        # [batch_size, num_agents]
        
        # 4. Embed actions and construct the autoregressive input sequence (shift actions right)
        action_embs = self.decoder.action_embedding(ordered_actions)    # [batch_size, num_agents, hidden_dim]
        start_tokens = self.start_action_emb.expand(batch_size, -1, -1) # [batch_size, 1, hidden_dim]
        prev_action_embeddings = torch.cat([start_tokens, action_embs[:, :-1, :]], dim=1) # Shift right
        
        # 5. Decode action logits using a causal mask to enforce step-by-step autoregression
        causal_mask = torch.triu(torch.ones(num_agents, num_agents, device=device), diagonal=1).bool()
        action_logits = self.decoder(target_agent_tokens, prev_action_embeddings, causal_mask=causal_mask)
        
        # 6. Compute action log probabilities and entropy
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        action_log_probs = dist.log_prob(ordered_actions)
        joint_action_log_prob = action_log_probs.sum(dim=-1) + perm_log_prob
        entropy = dist.entropy().mean()
        
        # 7. Centralized Value estimation
        values = self.critic(global_state)
        
        return joint_action_log_prob, entropy, values

    def select_actions(self, obs_n, deterministic=False):
        """
        Execution Phase (Rollout): Autoregressively sample actions one agent at a time 
        based on the dynamically generated Plackett-Luce execution ordering.
        """
        self.eval()
        with torch.no_grad():
            batch_size, num_agents, _ = obs_n.shape
            device = obs_n.device
            
            # 1. Encode observations
            agent_tokens = self.encoder(obs_n)  # [batch_size, num_agents, hidden_dim]
            
            # 2. Sample prioritization execution order
            permutation, _ = self.prioritizer(agent_tokens, deterministic=deterministic)
            
            # 3. Arrange agent tokens in selected execution order
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, num_agents)
            target_agent_tokens = agent_tokens[batch_indices, permutation]
            
            # 4. Initialize action embeddings sequence with the start token
            prev_action_embeddings = self.start_action_emb.expand(batch_size, 1, -1)
            sampled_actions = []
            
            # 5. Autoregressively decode actions agent-by-agent in the sampled order
            for step in range(num_agents):
                # Slice target agent tokens corresponding to current step
                curr_agent_tokens = target_agent_tokens[:, :step+1, :]
                
                # Predict action logits for the current agent
                action_logits = self.decoder(curr_agent_tokens, prev_action_embeddings)
                next_agent_logits = action_logits[:, -1, :]  # Grab logits for the current step
                
                # Sample action
                probs = torch.softmax(next_agent_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()
                
                sampled_actions.append(action)
                
                # Embed selected action and append to history sequence for next agent
                if step < num_agents - 1:
                    action_emb = self.decoder.action_embedding(action).unsqueeze(1)
                    prev_action_embeddings = torch.cat([prev_action_embeddings, action_emb], dim=1)

            # Reconstruct actions back to natural agent indexing (Agent 0, 1, ..., N)
            ordered_sampled_actions = torch.stack(sampled_actions, dim=1) # [batch_size, num_agents]
            
            # Scatter actions back to their original slots
            natural_actions = torch.zeros_like(ordered_sampled_actions)
            natural_actions.scatter_(1, permutation, ordered_sampled_actions)
            
            return natural_actions, permutation


class MATPolicy(PMATPolicy):
    """
    Standard Multi-Agent Transformer (MAT) Policy with a fixed/random order.
    """
    def __init__(self, obs_dim, action_dim, global_state_dim, hidden_dim=128):
        super(MATPolicy, self).__init__(obs_dim, action_dim, global_state_dim, hidden_dim)
        # Override the prioritizer to use the fixed ordering
        self.prioritizer = FixedPrioritizer()


class RoMATPolicy(PMATPolicy):
    """
    Role-based Multi-Agent Transformer (RoMAT) Policy.
    Concatenates a learnable role embedding to each agent's observation.
    """
    def __init__(self, obs_dim, action_dim, global_state_dim, num_agents=2, hidden_dim=128, role_dim=16):
        # Policy is initialized with obs_dim + role_dim input dimension
        super(RoMATPolicy, self).__init__(
            obs_dim=obs_dim + role_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            hidden_dim=hidden_dim
        )
        self.num_agents = num_agents
        self.role_embedding = nn.Embedding(num_agents, role_dim)

    def get_role_obs(self, obs_n):
        """
        obs_n: [batch_size, num_agents, obs_dim]
        Appends a unique role embedding to each agent's observation vector.
        """
        batch_size, num_agents, _ = obs_n.shape
        device = obs_n.device
        
        # Create role indices: [0, 1, ..., num_agents-1]
        role_indices = torch.arange(num_agents, device=device).unsqueeze(0).expand(batch_size, -1)
        role_embs = self.role_embedding(role_indices)  # [batch_size, num_agents, role_dim]
        
        # Concatenate observations with role embeddings
        return torch.cat([obs_n, role_embs], dim=-1)

    def evaluate_actions(self, obs_n, actions_n, global_state):
        role_obs = self.get_role_obs(obs_n)
        return super(RoMATPolicy, self).evaluate_actions(role_obs, actions_n, global_state)

    def select_actions(self, obs_n, deterministic=False):
        role_obs = self.get_role_obs(obs_n)
        return super(RoMATPolicy, self).select_actions(role_obs, deterministic=deterministic)


# --- Quick Test Rig ---
if __name__ == "__main__":
    print("Testing PMAT / MAT / RoMAT Policy Architectures...")
    batch_size = 4
    num_agents = 2
    obs_dim = 70
    action_dim = 4
    global_state_dim = num_agents * obs_dim
    
    dummy_obs = torch.randn(batch_size, num_agents, obs_dim)
    dummy_state = dummy_obs.view(batch_size, -1)
    
    for name, policy_class in [("PMAT", PMATPolicy), ("MAT", MATPolicy), ("RoMAT", RoMATPolicy)]:
        print(f"\n--- Testing {name} Policy ---")
        policy = policy_class(obs_dim=obs_dim, action_dim=action_dim, global_state_dim=global_state_dim)
        
        # Test selection
        actions, order = policy.select_actions(dummy_obs)
        print(f"  Actions shape: {actions.shape}, Order: {order[0].tolist()}")
        
        # Test evaluation
        log_probs, entropy, values = policy.evaluate_actions(dummy_obs, actions, dummy_state)
        print(f"  Evaluation - Log probs shape: {log_probs.shape}, Values shape: {values.shape}")
        
    print("\n✓ All modular MARL policies successfully verified!")
