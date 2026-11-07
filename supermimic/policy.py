"""
policy.py
---------
Tracking Policy Network for SuperMimic.
Maps (state, reference_target, tracking_error) -> motor actions.
Optimized via BPTT through the Differentiable World Model.
"""

import torch
import torch.nn as nn

class TrackingPolicy(nn.Module):
    def __init__(self, state_dim: int = 34, action_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        # Input features:
        # state (34) + target joint pos (12) + error (12) = 58
        in_dim = state_dim + action_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output actions scaled in [-1, 1]
        )

        # Joint position indices in the 34-dim state vector:
        # State layout:
        # 0..2: projected gravity (3)
        # 3: root z (1)
        # 4..6: base lin vel (3)
        # 7..9: base ang vel (3)
        # 10..21: 12 leg joint angles (12)
        # 22..33: 12 leg joint velocities (12)
        self.joint_pos_start = 10
        self.joint_pos_end = 22

    def forward(self, state: torch.Tensor, ref_target: torch.Tensor) -> torch.Tensor:
        """
        state: (B, 34)
        ref_target: (B, 12)
        """
        cur_joints = state[:, self.joint_pos_start:self.joint_pos_end]
        error = cur_joints - ref_target
        x = torch.cat([state, ref_target, error], dim=-1)
        action = self.net(x)
        return action

