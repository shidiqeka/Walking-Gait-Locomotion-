"""
world_model.py
--------------
Differentiable Neural World Model for SuperMimic.
Approximates robot multi-body physics: s_{t+1} = s_t + delta_s
Trained via supervised regression on simulator transitions.
Enables exact analytic gradient backpropagation through time (BPTT).
"""

import torch
import torch.nn as nn
import numpy as np

class DifferentiableWorldModel(nn.Module):
    def __init__(self, state_dim: int = 34, action_dim: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input projection
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Initialize output layer weights to small values for stable residual learning
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single-step transition:
          s_{t+1} = s_t + Delta s_t
        """
        x = torch.cat([state, action], dim=-1)
        delta_s = self.net(x)
        next_state = state + delta_s
        return next_state

    def rollout(self, initial_state: torch.Tensor, policy_net, ref_horizon: torch.Tensor) -> torch.Tensor:
        """
        Unroll dynamics through time for horizon H steps.
        initial_state: (B, state_dim)
        policy_net: callable actor network
        ref_horizon: (B, H, ref_dim)
        Returns: (B, H, state_dim)
        """
        batch_size, H, _ = ref_horizon.shape
        states = []
        cur_state = initial_state

        for t in range(H):
            ref_t = ref_horizon[:, t]
            action = policy_net(cur_state, ref_t)
            cur_state = self.forward(cur_state, action)
            states.append(cur_state)

        return torch.stack(states, dim=1)

