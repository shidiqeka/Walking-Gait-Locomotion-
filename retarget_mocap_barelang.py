#!/usr/bin/env python3
"""
retarget_mocap_barelang.py
--------------------------
Retargets NVIDIA AMP Humanoid Motion Capture dataset (CMU-based)
to the BarelangS Humanoid Robot (21 DOFs, 20 actuators).

Outputs:
  - data/mocap/barelang_mocap_walk.npz:
    Contains:
      - 'qpos': (N, 27) float32 trajectory
      - 'qvel': (N, 26) float32 velocities computed via MuJoCo differentiation
      - 'fps': 60
      - 'dt': 1/60.0
      - 'cycle_start': 18
      - 'cycle_end': 80 (62 frames = 1.033s gait period)
      - 'stride_length': forward distance per cycle
"""

import os
import numpy as np
import scipy.spatial.transform as sptr
import mujoco

def retarget_walk_mocap(
    raw_mocap_path: str = "data/mocap/amp_humanoid_walk.npy",
    xml_path: str = "barelangS_terrain_mujoco.xml",
    output_path: str = "data/mocap/barelang_mocap_walk.npz",
):
    print(f"Loading raw MoCap data from: {raw_mocap_path}")
    data = np.load(raw_mocap_path, allow_pickle=True).item()
    rot = data["rotation"]["arr"]             # (N, 15, 4) in [x, y, z, w]
    root_pos = data["root_translation"]["arr"] # (N, 3)
    n_frames = rot.shape[0]
    dt = 1.0 / 60.0

    print(f"Loading MuJoCo model from: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Scale factor: BarelangS leg length ~0.38m vs Human leg length ~0.83m
    scale = 0.38 / 0.83

    # Output arrays
    qpos_traj = np.zeros((n_frames, m.nq), dtype=np.float32)
    qvel_traj = np.zeros((n_frames, m.nv), dtype=np.float32)

    # Align starting position so robot starts at X=0, Y=0
    x0 = root_pos[0, 0]
    y0 = root_pos[0, 1]
    z_mean = np.mean(root_pos[:, 2])

    for f in range(n_frames):
        # 1. Root translation
        qpos_traj[f, 0] = (root_pos[f, 0] - x0) * scale
        qpos_traj[f, 1] = (root_pos[f, 1] - y0) * scale
        # Vertical motion: center around 0.0 (BarelangS zero-pose standing height)
        qpos_traj[f, 2] = (root_pos[f, 2] - z_mean) * scale

        # 2. Root orientation: Human Pelvis -> BarelangS Torso
        pelvis_q = rot[f, 0]  # [x, y, z, w]
        # MuJoCo freejoint expects [w, x, y, z]
        qpos_traj[f, 3] = pelvis_q[3]
        qpos_traj[f, 4] = pelvis_q[0]
        qpos_traj[f, 5] = pelvis_q[1]
        qpos_traj[f, 6] = pelvis_q[2]

        # 3. Upper body (Shoulders, Elbows, Head)
        # Head
        qpos_traj[f, 7] = 0.0  # head_yaw
        qpos_traj[f, 8] = 0.0  # head_pitch

        # Right Arm (MoCap node 3: right shoulder)
        r_arm_euler = sptr.Rotation.from_quat(rot[f, 3]).as_euler("yxz")
        qpos_traj[f, 9]  = 1.57 + r_arm_euler[0] * 0.45  # right_shoulder_pitch (arm swing)
        qpos_traj[f, 10] = -0.15                         # right_shoulder_roll (tucked)
        qpos_traj[f, 11] = -0.75                         # right_elbow (natural bend)

        # Left Arm (MoCap node 6: left shoulder)
        l_arm_euler = sptr.Rotation.from_quat(rot[f, 6]).as_euler("yxz")
        qpos_traj[f, 12] = 1.57 + l_arm_euler[0] * 0.45  # left_shoulder_pitch (counter swing)
        qpos_traj[f, 13] = 0.15                          # left_shoulder_roll
        qpos_traj[f, 14] = -0.75                         # left_elbow

        # 4. Right Leg (Thigh: 9, Shin: 10, Foot: 11)
        r_thigh_euler = sptr.Rotation.from_quat(rot[f, 9]).as_euler("zxy")
        qpos_traj[f, 15] = r_thigh_euler[0] * 0.5        # right_hip_yaw
        qpos_traj[f, 16] = r_thigh_euler[1]              # right_hip_roll
        qpos_traj[f, 17] = r_thigh_euler[2]              # right_hip_pitch

        r_knee_pitch = sptr.Rotation.from_quat(rot[f, 10]).as_euler("xyz")[1]
        qpos_traj[f, 18] = np.clip(r_knee_pitch, 0.0, 2.0)  # right_knee (flexion only)

        r_foot_euler = sptr.Rotation.from_quat(rot[f, 11]).as_euler("xyz")
        qpos_traj[f, 19] = r_foot_euler[1]               # right_ankle_pitch
        qpos_traj[f, 20] = r_foot_euler[0] * 0.6         # right_ankle_roll

        # 5. Left Leg (Thigh: 12, Shin: 13, Foot: 14)
        l_thigh_euler = sptr.Rotation.from_quat(rot[f, 12]).as_euler("zxy")
        qpos_traj[f, 21] = l_thigh_euler[0] * 0.5        # left_hip_yaw
        qpos_traj[f, 22] = l_thigh_euler[1]              # left_hip_roll
        qpos_traj[f, 23] = l_thigh_euler[2]              # left_hip_pitch

        l_knee_pitch = sptr.Rotation.from_quat(rot[f, 13]).as_euler("xyz")[1]
        qpos_traj[f, 24] = np.clip(l_knee_pitch, 0.0, 2.0)  # left_knee

        l_foot_euler = sptr.Rotation.from_quat(rot[f, 14]).as_euler("xyz")
        qpos_traj[f, 25] = l_foot_euler[1]               # left_ankle_pitch
        qpos_traj[f, 26] = l_foot_euler[0] * 0.6         # left_ankle_roll

    # Ground alignment: ensure the lowest foot point touches the ground plane (Z = 0)
    r_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_sole")
    l_sole_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_sole")

    min_foot_z = 1e9
    for f in range(n_frames):
        d.qpos[:] = qpos_traj[f]
        mujoco.mj_forward(m, d)
        rz = d.geom_xpos[r_sole_id, 2]
        lz = d.geom_xpos[l_sole_id, 2]
        min_foot_z = min(min_foot_z, rz, lz)

    # Shift root Z so minimum sole height is 0.005m (5mm contact margin)
    z_offset = 0.005 - min_foot_z
    qpos_traj[:, 2] += z_offset
    print(f"Applied vertical offset: {z_offset * 1000.0:.2f} mm to align foot soles with ground.")

    # Compute smooth joint and body velocities using mujoco.mj_differentiatePos
    for f in range(n_frames - 1):
        vel = np.zeros(m.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(m, vel, dt, qpos_traj[f], qpos_traj[f + 1])
        qvel_traj[f] = vel.astype(np.float32)
    # Last frame copies second to last
    qvel_traj[-1] = qvel_traj[-2]

    # Gait cycle metrics
    cycle_start = 18
    cycle_end = 80
    cycle_frames = cycle_end - cycle_start
    stride_len = float(qpos_traj[cycle_end, 0] - qpos_traj[cycle_start, 0])
    stride_dur = float(cycle_frames * dt)
    walking_speed = stride_len / stride_dur

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
        walking_speed=walking_speed,
    )

    print(f"\n[OK] Retargeting complete!")
    print(f"Saved dataset to: {output_path}")
    print(f"Frames: {n_frames} ({n_frames * dt:.2f} s)")
    print(f"Stride Length: {stride_len:.3f} m | Duration: {stride_dur:.3f} s")
    print(f"Walking Speed: {walking_speed:.3f} m/s ({walking_speed * 3.6:.2f} km/h)")
    print(f"Knee range (R): [{np.degrees(np.min(qpos_traj[:, 18])):.1f}°, {np.degrees(np.max(qpos_traj[:, 18])):.1f}°]")
    print(f"Hip Pitch range (R): [{np.degrees(np.min(qpos_traj[:, 17])):.1f}°, {np.degrees(np.max(qpos_traj[:, 17])):.1f}°]")

if __name__ == "__main__":
    retarget_walk_mocap()

