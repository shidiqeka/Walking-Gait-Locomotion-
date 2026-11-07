import os
import argparse
import math
import numpy as np
from stable_baselines3 import PPO
from barelang_env import BarelangSWalkEnv

def evaluate_policy(model_path="models/ppo_barelangS_best.zip", num_episodes_per_mode=3, max_steps=500):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return None

    print("=" * 80)
    print("        BARELANG S HUMANOID STEPPING POLICY EVALUATION")
    print("=" * 80)
    print(f"Evaluating Model : {model_path}")
    print(f"Steps per Test   : {max_steps} ({max_steps*0.02:.1f} seconds at 50 Hz)")
    print("=" * 80)

    env = BarelangSWalkEnv(training=False)
    model = PPO.load(model_path)

    test_modes = [
        ("Maju (Forward)",           [ 0.20,  0.00,  0.00]),
        ("Mundur (Backward)",        [-0.12,  0.00,  0.00]),
        ("Geser Kiri (Strafe Left)",  [ 0.00,  0.12,  0.00]),
        ("Geser Kanan (Strafe Right)",[ 0.00, -0.12,  0.00]),
        ("Rotasi Kiri (Turn CCW)",    [ 0.00,  0.00,  0.35]),
        ("Rotasi Kanan (Turn CW)",    [ 0.00,  0.00, -0.35]),
        ("Diam di Tempat (Stand)",    [ 0.00,  0.00,  0.00]),
    ]

    results_table = []
    all_survival = []
    all_speeds = []

    for mode_name, cmd in test_modes:
        mode_survivals = []
        mode_speeds = []
        mode_rewards = []

        for ep in range(num_episodes_per_mode):
            obs, _ = env.reset(seed=1000 + ep)
            env.set_command(*cmd)
            ep_reward = 0.0
            x_start = env.data.qpos[0]
            y_start = env.data.qpos[1]

            for step in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                if terminated or truncated:
                    break

            duration = (step + 1) * env.dt
            x_end = env.data.qpos[0]
            y_end = env.data.qpos[1]

            # Measure speed along commanded direction
            if abs(cmd[0]) > 0.05:
                speed = (x_end - x_start) / duration
            elif abs(cmd[1]) > 0.05:
                speed = (y_end - y_start) / duration
            elif abs(cmd[2]) > 0.05:
                speed = abs(info["act_wz"])
            else:
                speed = math.sqrt((x_end - x_start)**2 + (y_end - y_start)**2) / duration

            survived = (step >= max_steps - 1) and not terminated
            mode_survivals.append(1.0 if survived else 0.0)
            mode_speeds.append(speed)
            mode_rewards.append(ep_reward)

        survival_rate = float(np.mean(mode_survivals) * 100.0)
        mean_speed = float(np.mean(mode_speeds))
        mean_reward = float(np.mean(mode_rewards))

        results_table.append({
            "mode": mode_name,
            "cmd": cmd,
            "survival_rate": survival_rate,
            "speed": mean_speed,
            "reward": mean_reward
        })
        all_survival.extend(mode_survivals)
        all_speeds.extend(mode_speeds)

    # Print results
    print(f"\n{'Mode Pengujian':<28} | {'Command [Vx, Vy, Wz]':<20} | {'Survival Rate':<14} | {'Speed / Displacement':<20}")
    print("-" * 88)
    for res in results_table:
        cmd_str = f"[{res['cmd'][0]:+.2f}, {res['cmd'][1]:+.2f}, {res['cmd'][2]:+.2f}]"
        unit = "rad/s" if "Rotasi" in res['mode'] else "m/s"
        print(f"{res['mode']:<28} | {cmd_str:<20} | {res['survival_rate']:>6.1f} %        | {res['speed']:>+6.3f} {unit}")
    print("=" * 88)

    total_survival = float(np.mean(all_survival) * 100.0)
    print(f"\n>>> HASIL AKHIR EVALUASI:")
    print(f"    * TOTAL SUCCESS / SURVIVAL RATE : {total_survival:.2f} %")
    print("=" * 88)

    return total_survival

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BarelangS Policy")
    parser.add_argument("--model", type=str, default="models/ppo_barelangS_best.zip")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    import math
    evaluate_policy(model_path=args.model, num_episodes_per_mode=args.episodes)
