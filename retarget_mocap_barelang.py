#!/usr/bin/env python3
"""
retarget_mocap_barelang.py
--------------------------
Kinematic Retargeting Engine: Maps NVIDIA AMP Humanoid Motion Capture
(Walking & Running) to the BarelangS Humanoid Robot (21 DOFs, 20 Actuators).

Generates:
  - data/mocap/barelang_mocap_walk.npz (Walking: 0.71 m/s, 154 frames)
  - data/mocap/barelang_mocap_run.npz  (Running: 1.56 m/s, 82 frames)
"""

import os
import argparse
import numpy as np
import scipy.spatial.transform as sptr
import mujoco

def retarget_motion(
    raw_path: str,
    output_path: str,
    xml_path: str = "barelangS_terrain_mujoco.xml",
    is_running: bool = False
):
    print(f"\nProcessing {'RUN' if is_running else 'WALK'} MoCap: {raw_path}")
    data = np.load(raw_path, allow_pickle=True).item()
    rot = data["rotation"]["arr"]             # (N, 15, 4) in [x, y, z, w]
    root_pos = data["root_translation"]["arr"] # (N, 3)
    n_frames = rot.shape[0]
    dt = 1.0 / 60.0

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Scale factor: BarelangS leg length ~0.38m vs Human leg length ~0.83m
    scale = 0.38 / 0.83

    qpos_traj = np.zeros((n_frames, m.nq), dtype=np.float32)
    qvel_traj = np.zeros((n_frames, m.nv), dtype=np.float32)

    x0 = root_pos[0, 0]
    y0 = root_pos[0, 1]
    z_mean = np.mean(root_pos[:, 2])

    for f in range(n_frames):
        # 1. Root Translation
        qpos_traj[f, 0] = (root_pos[f, 0] - x0) * scale
        qpos_traj[f, 1] = (root_pos[f, 1] - y0) * scale
        qpos_traj[f, 2] = (root_pos[f, 2] - z_mean) * scale

        # 2. Pelvis Orientation
        pelvis_q = rot[f, 0] # [x, y, z, w]
        qpos_traj[f, 3] = pelvis_q[3] # w
        qpos_traj[f, 4] = pelvis_q[0] # x
        qpos_traj[f, 5] = pelvis_q[1] # y
        qpos_traj[f, 6] = pelvis_q[2] # z

        # 3. Arms
        r_arm_p = sptr.Rotation.from_quat(rot[f, 3]).as_euler("yxz")[0]
        l_arm_p = sptr.Rotation.from_quat(rot[f, 6]).as_euler("yxz")[0]
        arm_gain = 0.70 if is_running else 0.45
        elbow_angle = -1.25 if is_running else -0.75

        qpos_traj[f, 9]  = 1.57 + r_arm_p * arm_gain  # R shoulder pitch
        qpos_traj[f, 10] = -0.15                      # R shoulder roll
        qpos_traj[f, 11] = elbow_angle                # R elbow
        qpos_traj[f, 12] = 1.57 + l_arm_p * arm_gain  # L shoulder pitch
        qpos_traj[f, 13] = 0.15                       # L shoulder roll
        qpos_traj[f, 14] = elbow_angle                # L elbow

        # 4. Right Leg
        r_thigh = sptr.Rotation.from_quat(rot[f, 9]).as_euler("zxy")
        qpos_traj[f, 15] = r_thigh[0] * 0.5
        qpos_traj[f, 16] = r_thigh[1]
        qpos_traj[f, 17] = r_thigh[2]
        r_knee = sptr.Rotation.from_quat(rot[f, 10]).as_euler("xyz")[1]
        qpos_traj[f, 18] = np.clip(r_knee, 0.0, 2.1)
        r_foot = sptr.Rotation.from_quat(rot[f, 11]).as_euler("xyz")
        qpos_traj[f, 19] = r_foot[1]
        qpos_traj[f, 20] = r_foot[0] * 0.6

        # 5. Left Leg
        l_thigh = sptr.Rotation.from_quat(rot[f, 12]).as_euler("zxy")
        qpos_traj[f, 21] = l_thigh[0] * 0.5
        qpos_traj[f, 22] = l_thigh[1]
        qpos_traj[f, 23] = l_thigh[2]
        l_knee = sptr.Rotation.from_quat(rot[f, 13]).as_euler("xyz")[1]
        qpos_traj[f, 24] = np.clip(l_knee, 0.0, 2.1)
        l_foot = sptr.Rotation.from_quat(rot[f, 14]).as_euler("xyz")
        qpos_traj[f, 25] = l_foot[1]
        qpos_traj[f, 26] = l_foot[0] * 0.6

    # Ground alignment
    r_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_sole")
    l_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_sole")

    min_foot_z = 1e9
    for f in range(n_frames):
        d.qpos[:] = qpos_traj[f]
        mujoco.mj_forward(m, d)
        min_foot_z = min(min_foot_z, d.geom_xpos[r_sole_id, 2], d.geom_xpos[l_sole_id, 2])

    z_offset = 0.005 - min_foot_z
    qpos_traj[:, 2] += z_offset
    print(f"Applied vertical offset: {z_offset * 1000.0:.2f} mm to align soles with ground.")

    # Differentiate velocities via MuJoCo
    for f in range(n_frames - 1):
        vel = np.zeros(m.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(m, vel, dt, qpos_traj[f], qpos_traj[f + 1])
        qvel_traj[f] = vel.astype(np.float32)
    qvel_traj[-1] = qvel_traj[-2]

    # Stride parameters
    if is_running:
        cycle_start, cycle_end = 14, 60
    else:
        cycle_start, cycle_end = 18, 80

    cycle_frames = cycle_end - cycle_start
    stride_len = float(qpos_traj[cycle_end, 0] - qpos_traj[cycle_start, 0])
    stride_dur = float(cycle_frames * dt)
    speed = stride_len / stride_dur

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        qpos=qpos_traj,
        qvel=qvel_traj,
        fps=60,
        dt=dt,
        n_frames=n_frames,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        stride_length=stride_len,
        stride_duration=stride_dur,
        walking_speed=speed,
    )

    print(f"[OK] Saved {output_path}")
    print(f"  Frames: {n_frames} ({n_frames * dt:.2f} s) | Stride: {stride_len:.3f} m ({stride_dur:.3f} s)")
    print(f"  Speed: {speed:.3f} m/s ({speed * 3.6:.2f} km/h)")

if __name__ == "__main__":
    retarget_motion("data/mocap/amp_humanoid_walk.npy", "data/mocap/barelang_mocap_walk.npz", is_running=False)
    retarget_motion("data/mocap/amp_humanoid_run.npy",  "data/mocap/barelang_mocap_run.npz",  is_running=True)
