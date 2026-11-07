"""
mocap_dataset.py
----------------
Manages retargeted MoCap trajectories for SuperMimic training.
Matches the canonical 12-leg actuator ordering of BarelangS:
  [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll,
   right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll]
"""

import numpy as np
import torch
from typing import Tuple, List

LEG_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll"
]

# MuJoCo address mappings for BarelangS:
LEG_QPOS_ADR = [21, 22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20]
LEG_QVEL_ADR = [20, 21, 22, 23, 24, 25, 14, 15, 16, 17, 18, 19]
LEG_ACT_ID   = [12, 0, 1, 2, 3, 4, 13, 5, 6, 7, 8, 9]

class MocapDataset:
    def __init__(self, npz_path: str = "data/mocap/barelang_mocap_walk.npz", device: str = "cpu"):
        data = np.load(npz_path)
        self.qpos = data["qpos"].astype(np.float32)  # (N, 27)
        self.qvel = data["qvel"].astype(np.float32)  # (N, 26)
        self.n_frames = len(self.qpos)
        self.dt = float(data["dt"])
        self.fps = int(data["fps"])
        self.cycle_start = int(data.get("cycle_start", 18))
        self.cycle_end = int(data.get("cycle_end", 80))
        self.cycle_len = self.cycle_end - self.cycle_start
        self.stride_len = float(data.get("stride_length", 0.733))
        self.device = device

        self.leg_qpos_indices = LEG_QPOS_ADR
        self.leg_qvel_indices = LEG_QVEL_ADR

        # Canonical leg angles (N, 12)
        self.canonical_leg_qpos = self.qpos[:, self.leg_qpos_indices]
        self.canonical_leg_qvel = self.qvel[:, self.leg_qvel_indices]

        # Convert to torch tensors
        self.t_qpos = torch.tensor(self.qpos, device=device)
        self.t_qvel = torch.tensor(self.qvel, device=device)
        self.t_leg_qpos = torch.tensor(self.canonical_leg_qpos, device=device)
        self.t_leg_qvel = torch.tensor(self.canonical_leg_qvel, device=device)

    def sample_frame(self, loop_only: bool = True) -> int:
        """Sample a random reference frame index."""
        if loop_only:
            return np.random.randint(self.cycle_start, self.cycle_end)
        return np.random.randint(0, self.n_frames)

    def get_reference_pose(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns full (qpos, qvel) for a single frame."""
        idx = frame_idx % self.n_frames
        return self.qpos[idx].copy(), self.qvel[idx].copy()

    def get_leg_target(self, frame_idx: int) -> np.ndarray:
        """Returns 12 leg joint target angles for frame_idx in canonical order."""
        idx = frame_idx % self.n_frames
        return self.canonical_leg_qpos[idx].copy()

    def get_reference_horizon(self, frame_idx: int, horizon: int) -> torch.Tensor:
        """
        Returns reference joint angles for the next `horizon` frames.
        Shape: (horizon, 12) in canonical leg order.
        Handles cyclic wrapping seamlessly.
        """
        indices = []
        cur = frame_idx
        for _ in range(horizon):
            if cur >= self.cycle_end:
                cur = self.cycle_start + (cur - self.cycle_end) % self.cycle_len
            indices.append(cur)
            cur += 1

        return self.t_leg_qpos[indices]  # (horizon, 12)
