"""
Comprehensive Evaluation Script for Barelang S v2
Evaluates: terrain locomotion + push recovery + standing stability
"""
import os
import argparse
import numpy as np
import math
from stable_baselines3 import PPO
from barelang_terrain_env_v2 import BarelangSTerrainEnvV2

TERRAINS = ["rough", "wavy", "slopes", "mixed", "flat"]
LOCOMOTION_TESTS = [
    ("Maju (Forward)",    np.array([0.15, 0.0, 0.0], dtype=np.float32)),
    ("Mundur (Backward)", np.array([-0.12, 0.0, 0.0], dtype=np.float32)),
    ("Strafe Kiri",       np.array([0.0, 0.10, 0.0], dtype=np.float32)),
    ("Strafe Kanan",      np.array([0.0, -0.10, 0.0], dtype=np.float32)),
    ("Rotasi CCW",        np.array([0.0, 0.0, 0.35], dtype=np.float32)),
    ("Rotasi CW",         np.array([0.0, 0.0, -0.35], dtype=np.float32)),
]
PUSH_DIRECTIONS = [
    ("Dorong Depan",  [5.0, 0.0]),
    ("Dorong Belakang", [-5.0, 0.0]),
    ("Dorong Kiri",   [0.0, 5.0]),
    ("Dorong Kanan",  [0.0, -5.0]),
]


def evaluate(model_path, difficulty=0.4, steps_per_test=300):
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return

    print("=" * 90)
    print("    BARELANG S v2 - EVALUASI KOMPREHENSIF (TERRAIN + PUSH RECOVERY)")
    print("=" * 90)
    print(f"Model     : {model_path}")
    print(f"Difficulty: {difficulty}")
    print(f"Steps     : {steps_per_test} (~{steps_per_test * 0.02:.1f}s per test)")
    print("=" * 90)

    model = PPO.load(model_path)

    # ============================================================
    # PART 1: TERRAIN LOCOMOTION
    # ============================================================
    print("\n--- BAGIAN 1: LOKOMOTION TERRAIN ---")
    print(f"{'Terrain':<8} | {'Gerakan':<18} | {'Status':<9} | {'Dist X':<10} | {'Dist Y':<10} | {'MaxTilt':<10} | {'Sag':<8}")
    print("-" * 90)

    total_loco = 0
    survived_loco = 0
    terrain_stats = {}

    for terrain in TERRAINS:
        env = BarelangSTerrainEnvV2(terrain_type=terrain, terrain_difficulty=difficulty, training=False)
        terrain_stats[terrain] = {"survived": 0, "total": 0, "fwd": 0.0, "bwd": 0.0}

        for mode_name, cmd in LOCOMOTION_TESTS:
            total_loco += 1
            terrain_stats[terrain]["total"] += 1
            env.generate_terrain(mode=terrain, difficulty=difficulty, flat_radius=0.20)
            obs, _ = env.reset()
            env.set_command(cmd[0], cmd[1], cmd[2])
            env.target_v = cmd.copy()
            obs = env._get_obs()
            start_pos = env.data.qpos[:3].copy()
            max_tilt = 0.0
            max_sag = 0.0
            survived = True

            for step in range(steps_per_test):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)

                grav = obs[3:6]
                tilt = np.degrees(np.arccos(np.clip(-grav[2], -1.0, 1.0)))
                max_tilt = max(max_tilt, tilt)
                sag = abs(info.get("rel_torso_z", 0.418) - env.nominal_torso_height)
                max_sag = max(max_sag, sag)

                if term:
                    survived = False
                    break

            end_pos = env.data.qpos[:3]
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]

            if survived:
                survived_loco += 1
                terrain_stats[terrain]["survived"] += 1
                status = "SURVIVED"
            else:
                status = f"FELL@{step}"

            if "Forward" in mode_name:
                terrain_stats[terrain]["fwd"] = dx
            elif "Backward" in mode_name:
                terrain_stats[terrain]["bwd"] = dx

            print(f"{terrain:<8} | {mode_name:<18} | {status:<9} | {dx:+9.3f}  | {dy:+9.3f}  | {max_tilt:6.1f} deg | {max_sag:6.3f}m")

        env.close()
        print("-" * 90)

    # ============================================================
    # PART 2: PUSH RECOVERY
    # ============================================================
    print("\n--- BAGIAN 2: UJI KETAHANAN DORONG (PUSH RECOVERY) ---")
    print(f"{'Arah Dorong':<18} | {'Kekuatan':<10} | {'Status':<12} | {'Recovery Time':<14} | {'MaxTilt':<10}")
    print("-" * 70)

    push_passed = 0
    push_total = 0

    env = BarelangSTerrainEnvV2(terrain_type="flat", terrain_difficulty=0.0, training=False)

    for push_name, push_vec in PUSH_DIRECTIONS:
        for force_mult in [1.0, 1.5]:
            push_total += 1
            env.generate_terrain(mode="flat", difficulty=0.0)
            obs, _ = env.reset()
            env.set_command(0.0, 0.0, 0.0)
            env.target_v = np.zeros(3, dtype=np.float32)
            obs = env._get_obs()
            fallen = False

            # Stabilize for 50 steps
            for _ in range(50):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, _, _ = env.step(action)
                if term:
                    fallen = True
                    break

            if not fallen:
                # Apply push
                force = np.array(push_vec) * force_mult
                env.push_force[:2] = force
                env.push_steps = 10
                env.push_active = True
                force_mag = np.linalg.norm(force)

                recovery_step = -1
                max_tilt = 0.0
                for step in range(200):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, _, _ = env.step(action)
                    grav = obs[3:6]
                    tilt = np.degrees(np.arccos(np.clip(-grav[2], -1.0, 1.0)))
                    max_tilt = max(max_tilt, tilt)

                    if term:
                        fallen = True
                        break

                    # Check if recovered (tilt < 5 degrees)
                    if recovery_step < 0 and tilt < 5.0 and step > 15:
                        recovery_step = step

                if not fallen:
                    push_passed += 1
                    rec_time = f"{recovery_step * 0.02:.2f}s" if recovery_step > 0 else "N/A"
                    print(f"{push_name:<18} | {force_mag:6.1f} N  | {'RECOVERED':<12} | {rec_time:<14} | {max_tilt:6.1f} deg")
                else:
                    print(f"{push_name:<18} | {force_mag:6.1f} N  | {'JATUH@' + str(step):<12} | {'---':<14} | {max_tilt:6.1f} deg")
            else:
                print(f"{push_name:<18} | {'N/A':>6}    | {'PRE-FALL':<12} | {'---':<14} | {'---':>6}")

    env.close()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 90)
    print("                      RINGKASAN EVALUASI v2")
    print("=" * 90)

    loco_rate = (survived_loco / total_loco) * 100.0
    push_rate = (push_passed / push_total) * 100.0

    print(f"\nLOKOMOTION TERRAIN:")
    print(f"  Total: {survived_loco}/{total_loco} ({loco_rate:.1f}% Survival)")
    for t, s in terrain_stats.items():
        r = (s["survived"] / s["total"]) * 100.0
        print(f"    {t.upper():<7}: {r:5.1f}% | Maju: {s['fwd']:+.3f}m | Mundur: {s['bwd']:+.3f}m")

    print(f"\nPUSH RECOVERY:")
    print(f"  Total: {push_passed}/{push_total} ({push_rate:.1f}% Recovery Rate)")

    overall = (loco_rate * 0.6 + push_rate * 0.4)
    print(f"\nSKOR KESELURUHAN: {overall:.1f}% (60% Lokomotion + 40% Push Recovery)")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_barelangS_v2_best.zip")
    parser.add_argument("--difficulty", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    evaluate(args.model, args.difficulty, args.steps)
