"""
Dedicated User Goal Verification Script
Tests:
1. Straight Forward (Normal & Fast Trot)
2. Straight Backward
3. Pure Lateral Strafe (Left & Right)
4. Pure In-Place Rotation (CCW & CW) - Zero translation drift
5. Standing Posture & Foot Flatness
"""
import math
import numpy as np
from stable_baselines3 import PPO
from barelang_terrain_env_v2 import BarelangSTerrainEnvV2


def run_benchmark(model_path="models/ppo_barelangS_v2_best.zip", steps=300):
    print("=" * 95)
    print("           BARELANG S v2 - BENCHMARK KESESUAIAN TARGET USER (/GOAL)")
    print("=" * 95)
    print(f"Model: {model_path} | Steps: {steps} ({steps * 0.02:.1f}s)")
    print("=" * 95)

    model = PPO.load(model_path)
    env = BarelangSTerrainEnvV2(training=False, terrain_type="flat", terrain_difficulty=0.0)

    test_scenarios = [
        # Name, cmd [vx, vy, wz], description
        ("1. Jalan Lurus Depan (Normal)",  [ 0.16,  0.00,  0.00], "Pure straight forward walk"),
        ("2. Jalan Lurus Depan (Trot/Fast)",[ 0.28,  0.00,  0.00], "Fast walk / trot forward"),
        ("3. Jalan Lurus Belakang",         [-0.12,  0.00,  0.00], "Pure straight backward walk"),
        ("4. Jalan Menyamping Kiri",        [ 0.00,  0.12,  0.00], "Pure lateral strafe left"),
        ("5. Jalan Menyamping Kanan",       [ 0.00, -0.12,  0.00], "Pure lateral strafe right"),
        ("6. Rotasi di Tempat (CCW)",       [ 0.00,  0.00,  0.35], "In-place rotation counter-clockwise"),
        ("7. Rotasi di Tempat (CW)",        [ 0.00,  0.00, -0.35], "In-place rotation clockwise"),
        ("8. Diam Tegak (Natural Stance)",  [ 0.00,  0.00,  0.00], "Upright standing stability"),
    ]

    print(f"\n{'Skenario Uji':<32} | {'dX (m)':<8} | {'dY (m)':<8} | {'Yaw Drift':<10} | {'Torso Pitch':<11} | {'Foot Tilt':<10} | {'Status'}")
    print("-" * 95)

    for name, cmd, desc in test_scenarios:
        env.generate_terrain(mode="flat", difficulty=0.0)
        obs, _ = env.reset(seed=42)
        env.set_command(*cmd)
        env.target_v = np.array(cmd, dtype=np.float32)
        obs = env._get_obs()

        start_x = env.data.qpos[0]
        start_y = env.data.qpos[1]
        start_w, start_qx, start_qy, start_qz = env.data.qpos[3:7]
        start_yaw = math.atan2(2 * (start_w * start_qz + start_qx * start_qy), 1 - 2 * (start_qy**2 + start_qz**2))

        torso_pitches = []
        foot_tilts = []
        survived = True

        for s in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)

            # Torso pitch
            w, qx, qy, qz = env.data.qpos[3:7]
            pitch_rad = math.atan2(2 * (w * qy - qx * qz), 1 - 2 * (qy**2 + qz**2))
            torso_pitches.append(math.degrees(pitch_rad))

            # Foot tilt
            rf_mat = env.data.xmat[env.r_foot_bid].reshape(3, 3)
            lf_mat = env.data.xmat[env.l_foot_bid].reshape(3, 3)
            rf_tilt = math.degrees(math.acos(np.clip(rf_mat[2, 2], -1.0, 1.0)))
            lf_tilt = math.degrees(math.acos(np.clip(lf_mat[2, 2], -1.0, 1.0)))
            foot_tilts.append((rf_tilt + lf_tilt) * 0.5)

            if term:
                survived = False
                break

        end_x = env.data.qpos[0]
        end_y = env.data.qpos[1]
        dx = end_x - start_x
        dy = end_y - start_y

        w, qx, qy, qz = env.data.qpos[3:7]
        end_yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        yaw_diff = (end_yaw - start_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_drift_deg = math.degrees(yaw_diff)

        avg_pitch = np.mean(torso_pitches) if torso_pitches else 0.0
        avg_foot_tilt = np.mean(foot_tilts) if foot_tilts else 0.0

        status_str = "LULUS (STABIL)" if survived else f"JATUH @ {s}"
        yaw_str = f"{yaw_drift_deg:+6.1f}°" if abs(cmd[2]) < 0.01 else f"Rotated"
        print(f"{name:<32} | {dx:+7.3f}m | {dy:+7.3f}m | {yaw_str:<10} | {avg_pitch:+6.2f}°    | {avg_foot_tilt:6.2f}°    | {status_str}")

    env.close()
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
