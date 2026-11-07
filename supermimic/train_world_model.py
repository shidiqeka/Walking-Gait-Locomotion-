"""
train_world_model.py
--------------------
Trains the Differentiable World Model on transitions collected from MuJoCo.
Predicts state residual: \Delta s = s_{t+1} - s_t
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from supermimic.world_model import DifferentiableWorldModel

def train_world_model(
    data_path: str = "data/mocap/world_model_transitions.npz",
    save_path: str = "models/supermimic_world_model.pt",
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu"
):
    print(f"Loading transitions from: {data_path}")
    raw = np.load(data_path)
    states = raw["states"]
    actions = raw["actions"]
    next_states = raw["next_states"]

    n_samples = len(states)
    print(f"Dataset: {n_samples} transitions, State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

    # Train / Val split (90% / 10%)
    indices = np.random.permutation(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    s_train = torch.tensor(states[train_idx], dtype=torch.float32)
    a_train = torch.tensor(actions[train_idx], dtype=torch.float32)
    ns_train = torch.tensor(next_states[train_idx], dtype=torch.float32)

    s_val = torch.tensor(states[val_idx], dtype=torch.float32, device=device)
    a_val = torch.tensor(actions[val_idx], dtype=torch.float32, device=device)
    ns_val = torch.tensor(next_states[val_idx], dtype=torch.float32, device=device)

    train_ds = TensorDataset(s_train, a_train, ns_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = DifferentiableWorldModel(state_dim=states.shape[1], action_dim=actions.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print(f"\nTraining World Model for {epochs} epochs on {device}...")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for b_s, b_a, b_ns in train_loader:
            b_s, b_a, b_ns = b_s.to(device), b_a.to(device), b_ns.to(device)

            pred_ns = model(b_s, b_a)
            loss = criterion(pred_ns, b_ns)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(s_val, a_val)
            val_loss = criterion(pred_val, ns_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "state_dim": states.shape[1],
                "action_dim": actions.shape[1],
                "val_loss": val_loss,
            }, save_path)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print(f"\n[OK] Trained World Model saved to: {save_path} (Best Val MSE: {best_val_loss:.6f})")

if __name__ == "__main__":
    train_world_model()

