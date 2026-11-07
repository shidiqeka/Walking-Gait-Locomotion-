"""
train_policy_bptt.py
--------------------
SuperMimic Policy Trainer using Backpropagation Through Time (BPTT).
Rolls out the policy through the frozen Differentiable World Model for horizon H,
computing exact analytic gradients to track the MoCap reference walk.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from supermimic.mocap_dataset import MocapDataset, LEG_QPOS_ADR, LEG_QVEL_ADR
from supermimic.world_model import DifferentiableWorldModel
from supermimic.policy import TrackingPolicy

def train_supermimic_bptt(
    world_model_path: str = "models/supermimic_world_model.pt",
    mocap_path: str = "data/mocap/barelang_mocap_walk.npz",
    save_path: str = "models/supermimic_policy.pt",
    horizon: int = 8,
    batch_size: int = 128,
    train_steps: int = 1000,
    lr: float = 3e-4,
    device: str = "cpu"
):
    print("=" * 70)
    print(" SuperMimic: BPTT Policy Optimization")
    print("=" * 70)

    # 1. Load Mocap dataset
    mocap = MocapDataset(mocap_path, device=device)

    # 2. Load World Model and Freeze weights
    print(f"Loading World Model from: {world_model_path}")
    checkpoint = torch.load(world_model_path, map_location=device, weights_only=True)
    world_model = DifferentiableWorldModel(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"]
    ).to(device)
    world_model.load_state_dict(checkpoint["model_state_dict"])
    world_model.eval()
    for param in world_model.parameters():
        param.requires_grad = False

    # 3. Initialize Tracking Policy
    policy = TrackingPolicy(state_dim=checkpoint["state_dim"], action_dim=checkpoint["action_dim"]).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # Tracking weights
    w_pos = 10.0   # Joint angle tracking weight
    w_vel = 0.5    # Joint velocity tracking weight
    w_upright = 5.0 # Keep torso upright (projected gravity ~ [0, 0, -1])
    w_act = 0.05   # Action smoothness / regularization

    target_gravity = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0)

    print(f"\nTraining Tracking Policy for {train_steps} BPTT steps (Horizon H={horizon})...")

    # Pre-build state buffer for fast sampling
    transitions_data = np.load("data/mocap/world_model_transitions.npz")
    all_states = torch.tensor(transitions_data["states"], dtype=torch.float32, device=device)
    n_states = len(all_states)

    best_loss = float("inf")

    for step in range(1, train_steps + 1):
        # Sample random initial states from buffer
        idx = torch.randint(0, n_states, (batch_size,))
        cur_state = all_states[idx]  # (B, 34)

        # Sample starting frames for reference horizons
        batch_ref_horizons = []
        for _ in range(batch_size):
            start_f = mocap.sample_frame()
            ref_h = mocap.get_reference_horizon(start_f, horizon)  # (H, 12)
            batch_ref_horizons.append(ref_h)
        ref_horizons = torch.stack(batch_ref_horizons, dim=0).to(device)  # (B, H, 12)

        total_loss = 0.0
        pos_err_sum = 0.0
        upright_err_sum = 0.0

        # Unroll BPTT through World Model
        for t in range(horizon):
            ref_q_t = ref_horizons[:, t]  # (B, 12)
            action = policy(cur_state, ref_q_t)  # (B, 12)

            # Predict next physical state via Differentiable World Model
            next_state = world_model(cur_state, action)

            # Losses at time t:
            # Joint tracking error (indices 10..22 are leg joint angles)
            pred_q = next_state[:, 10:22]
            l_pos = torch.mean((pred_q - ref_q_t) ** 2)

            # Upright loss (indices 0..3 are projected gravity)
            pred_g = next_state[:, 0:3]
            l_upright = torch.mean((pred_g - target_gravity) ** 2)

            # Action effort penalty
            l_act = torch.mean(action ** 2)

            step_loss = w_pos * l_pos + w_upright * l_upright + w_act * l_act
            total_loss += step_loss
            pos_err_sum += l_pos.item()
            upright_err_sum += l_upright.item()

            cur_state = next_state

        loss = total_loss / horizon

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "step": step,
                "best_loss": best_loss,
            }, save_path)

        if step % 100 == 0 or step == 1 or step == train_steps:
            avg_pos = np.degrees(np.sqrt(pos_err_sum / horizon))
            print(
                f"Step {step:4d}/{train_steps} | Loss: {loss.item():.4f} | "
                f"Avg Joint Error: {avg_pos:.2f}° | Best: {best_loss:.4f}"
            )

    print(f"\n[OK] SuperMimic Policy trained and saved to: {save_path} (Best Loss: {best_loss:.4f})")

if __name__ == "__main__":
    train_supermimic_bptt()

