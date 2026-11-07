import os
import math
import numpy as np
from stable_baselines3 import PPO
from barelang_env import BarelangSWalkEnv

def verify_all_metrics(model_path="models/ppo_barelangS_best.zip"):
    print("=" * 85)
    print("      ANALISIS MENDALAM DINAMIKA & KELURUSAN JALAN BARELANG S")
    print("=" * 85)
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} tidak ditemukan.")
        return
        
    env = BarelangSWalkEnv(training=False)
    model = PPO.load(model_path)
    
    # Mode pengujian dengan target verifikasi spesifik sesuai permintaan user
    tests = [
        {
            "name": "1. Jalan Maju Lurus (Forward)",
            "cmd": [0.20, 0.0, 0.0],
            "drift_axis": "Y (Samping)",
            "primary_axis": "X (Maju)",
        },
        {
            "name": "2. Jalan Mundur Lurus (Backward)",
            "cmd": [-0.12, 0.0, 0.0],
            "drift_axis": "Y (Samping)",
            "primary_axis": "X (Mundur)",
        },
        {
            "name": "3. Jalan Samping Kiri (Strafe Left)",
            "cmd": [0.0, 0.12, 0.0],
            "drift_axis": "X (Maju/Mundur)",
            "primary_axis": "Y (Kiri)",
        },
        {
            "name": "4. Jalan Samping Kanan (Strafe Right)",
            "cmd": [0.0, -0.12, 0.0],
            "drift_axis": "X (Maju/Mundur)",
            "primary_axis": "Y (Kanan)",
        },
        {
            "name": "5. Rotasi Kiri di Tempat (Turn CCW)",
            "cmd": [0.0, 0.0, 0.35],
            "drift_axis": "XY Total Drift",
            "primary_axis": "Yaw Rotation",
        },
        {
            "name": "6. Rotasi Kanan di Tempat (Turn CW)",
            "cmd": [0.0, 0.0, -0.35],
            "drift_axis": "XY Total Drift",
            "primary_axis": "Yaw Rotation",
        },
    ]

    report = []
    
    for t in tests:
        obs, _ = env.reset(seed=42)
        env.set_command(*t["cmd"])
        
        x_init = env.data.qpos[0]
        y_init = env.data.qpos[1]
        
        max_foot_heights = []
        torso_roll_angles = []
        torso_z_heights = []
        
        steps = 500  # 10 detik penuh
        fell = False
        
        for s in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Catat parameter dinamika
            z_left_foot = env.data.site_xpos[env.left_foot_site_id, 2]
            z_right_foot = env.data.site_xpos[env.right_foot_site_id, 2]
            max_foot_heights.append(max(z_left_foot, z_right_foot))
            
            # Roll torso (orientasi roll dalam derajat)
            roll_deg = math.degrees(env.data.qpos[4])  # qpos[4] approx roll in small angles or rot mat
            # Lebih presisi dari rot matrix
            rot_mat = env.data.xmat[env.torso_body_id].reshape(3, 3)
            r_roll = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
            torso_roll_angles.append(r_roll)
            torso_z_heights.append(env.data.qpos[2])
            
            if terminated:
                fell = True
                break
                
        x_final = env.data.qpos[0]
        y_final = env.data.qpos[1]
        dx = x_final - x_init
        dy = y_final - y_init
        total_drift_xy = math.sqrt(dx**2 + dy**2)
        
        if "Forward" in t["name"] or "Backward" in t["name"]:
            primary_disp = dx
            drift = abs(dy)
            straightness_ratio = (abs(primary_disp) / (abs(primary_disp) + drift + 1e-6)) * 100
        elif "Strafe" in t["name"]:
            primary_disp = dy
            drift = abs(dx)
            straightness_ratio = (abs(primary_disp) / (abs(primary_disp) + drift + 1e-6)) * 100
        else: # Rotation
            primary_disp = info["act_wz"]
            drift = total_drift_xy
            straightness_ratio = 100.0 if drift < 0.05 else max(0.0, 100.0 - drift * 200)

        # Clearance kaki di atas tanah (posisi site telapak kaki saat diam di lantai adalah 0.007 m)
        step_clearance_cm = (np.percentile(max_foot_heights, 95) - 0.007) * 100
        max_roll_lean_deg = np.max(np.abs(torso_roll_angles))
        mean_z = np.mean(torso_z_heights)
        
        report.append({
            "name": t["name"],
            "fell": fell,
            "duration": (s + 1) * env.dt,
            "primary_disp": primary_disp,
            "drift": drift,
            "drift_axis": t["drift_axis"],
            "straightness": straightness_ratio,
            "step_clearance_cm": step_clearance_cm,
            "max_roll_lean_deg": max_roll_lean_deg,
            "mean_z": mean_z
        })

    print(f"\n{'Mode Uji':<35} | {'Jarak Utama':<12} | {'Drift / Melenceng':<16} | {'Kelurusan':<10} | {'Tinggi Kaki':<11} | {'Status'}")
    print("-" * 105)
    for r in report:
        status = "JATUH" if r["fell"] else "STABIL (100%)"
        print(f"{r['name']:<35} | {r['primary_disp']:>+8.2f} m  | {r['drift']:>6.3f} m ({r['drift_axis'][:3]}) | {r['straightness']:>6.1f} %   | {r['step_clearance_cm']:>5.1f} cm   | {status}")
        
    print("=" * 105)
    print("\nRingkasan Lean Torso & Ketinggian Langkah:")
    for r in report:
        print(f" - {r['name']:<32}: Dynamic Torso Lean = {r['max_roll_lean_deg']:.1f}° | Torso Height Z = {r['mean_z']:.3f} m")
    print("=" * 105)

if __name__ == "__main__":
    verify_all_metrics()
