"""
Terrain Locomotion Comprehensive Evaluation Script for Barelang S
Evaluates trained models across procedural terrain modes (Rough, Wavy, Slopes, Mixed, Flat).
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from barelang_terrain_env import BarelangSTerrainEnv

TERRAINS = ["rough", "wavy", "slopes", "mixed", "flat"]
TEST_MODES = [
    ("Maju (Forward)", np.array([0.15, 0.0, 0.0], dtype=np.float32)),
    ("Mundur (Backward)", np.array([-0.12, 0.0, 0.0], dtype=np.float32)),
    ("Strafe Kiri", np.array([0.0, 0.10, 0.0], dtype=np.float32)),
    ("Strafe Kanan", np.array([0.0, -0.10, 0.0], dtype=np.float32)),
    ("Rotasi CCW", np.array([0.0, 0.0, 0.35], dtype=np.float32)),
    ("Rotasi CW", np.array([0.0, 0.0, -0.35], dtype=np.float32)),
]

def evaluate_model(model_path, difficulty=0.6, steps_per_test=400):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print("=" * 80)
    print(f"       BARELANG S PROCEDURAL TERRAIN MULTI-MODE EVALUATION")
    print("=" * 80)
    print(f"Model       : {model_path}")
    print(f"Difficulty  : {difficulty}")
    print(f"Test Length : {steps_per_test} steps (~{steps_per_test*0.02:.1f}s) per mode")
    print("=" * 80)

    model = PPO.load(model_path)
    total_tests = 0
    total_survived = 0

    print(f"{'Terrain':<8} | {'Gerakan':<18} | {'Status':<9} | {'Dist X (m)':<10} | {'Dist Y (m)':<10} | {'Max Tilt':<10} | {'Local Sag':<10}")
    print("-" * 88)

    summary_by_terrain = {}

    for terrain in TERRAINS:
        env = BarelangSTerrainEnv(terrain_type=terrain, terrain_difficulty=difficulty, training=False)
        env.unwrapped.training = False
        summary_by_terrain[terrain] = {"total": 0, "survived": 0, "fwd_dist": 0.0, "bwd_dist": 0.0}

        for mode_name, cmd in TEST_MODES:
            total_tests += 1
            summary_by_terrain[terrain]["total"] += 1

            env.unwrapped.generate_terrain(mode=terrain, difficulty=difficulty, flat_radius=0.20)
            obs, _ = env.reset()
            env.unwrapped.set_command(cmd[0], cmd[1], cmd[2])
            env.unwrapped.target_v = cmd.copy()
            obs = env.unwrapped._get_obs()
            start_pos = env.unwrapped.data.qpos[:3].copy()
            max_tilt_deg = 0.0
            max_sag = 0.0
            survived = True

            for step in range(steps_per_test):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                # Track torso tilt from projected gravity
                grav = obs[3:6]
                tilt = np.arccos(np.clip(-grav[2], -1.0, 1.0)) * 180.0 / np.pi
                if tilt > max_tilt_deg:
                    max_tilt_deg = tilt

                # Track height relative to terrain
                rel_h = info.get("rel_torso_z", 0.418)
                sag = abs(rel_h - env.unwrapped.nominal_torso_height)
                if sag > max_sag:
                    max_sag = sag

                if terminated:
                    survived = False
                    break

            end_pos = env.unwrapped.data.qpos[:3]
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]

            if survived:
                total_survived += 1
                summary_by_terrain[terrain]["survived"] += 1
                status_str = "SURVIVED"
            else:
                status_str = f"FELL@{step}"

            if "Forward" in mode_name:
                summary_by_terrain[terrain]["fwd_dist"] = dx
            elif "Backward" in mode_name:
                summary_by_terrain[terrain]["bwd_dist"] = dx

            print(f"{terrain:<8} | {mode_name:<18} | {status_str:<9} | {dx:+9.3f}  | {dy:+9.3f}  | {max_tilt_deg:6.1f} deg  | {max_sag:7.3f} m")

        env.close()
        print("-" * 88)

    print("=" * 80)
    print("                    RINGKASAN HASIL EVALUASI TERRAIN")
    print("=" * 80)
    overall_rate = (total_survived / total_tests) * 100.0
    print(f"Total Uji Coba: {total_tests} skenario")
    print(f"Total Berhasil: {total_survived} / {total_tests} ({overall_rate:.1f}% Survival Rate)\n")

    for terrain, stat in summary_by_terrain.items():
        rate = (stat["survived"] / stat["total"]) * 100.0
        print(f"  * {terrain.upper():<7}: Survival {rate:5.1f}% | Maju: {stat['fwd_dist']:+0.3f} m | Mundur: {stat['bwd_dist']:+0.3f} m")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Barelang S on Procedural Terrains")
    parser.add_argument("--model", type=str, default="models/ppo_barelangS_terrain_best.zip")
    parser.add_argument("--difficulty", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=350)
    args = parser.parse_args()

    evaluate_model(args.model, args.difficulty, args.steps)
