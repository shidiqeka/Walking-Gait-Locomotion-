"""
Barelang S Terrain Training v2 - From Scratch
=============================================
Long-duration training pipeline designed for training from zero:
- 5M+ timesteps for convergence
- Larger network (256x256) for complex balance behaviors
- Curriculum learning with automatic difficulty scaling
- Comprehensive evaluation with push recovery tests
- Periodic checkpointing and logging
"""
import os
import sys
import argparse
import time
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from barelang_terrain_env_v2 import BarelangSTerrainEnvV2


class TerrainEvalCallbackV2(BaseCallback):
    """
    Comprehensive evaluation callback with push recovery testing.
    Tests across multiple terrains AND push recovery scenarios.
    """
    def __init__(self, eval_freq=25000, initial_best_score=-float("inf"), verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_score = initial_best_score
        self.eval_env = None
        self.start_time = None

    def _on_training_start(self) -> None:
        self.eval_env = BarelangSTerrainEnvV2(
            training=False, terrain_type="flat", terrain_difficulty=0.4
        )
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.model.n_envs:
            survivals = []
            fwd_speeds = []
            bwd_speeds = []
            strafe_speeds = []
            turn_speeds = []
            turn_drifts = []
            tilt_maxs = []

            # Locomotion tests across terrains matching user requirements
            locomotion_suite = [
                ("Flat Straight Forward",  "flat",   [ 0.18,  0.00,  0.00]),
                ("Flat Fast Trot/Run",     "flat",   [ 0.30,  0.00,  0.00]),
                ("Flat Straight Backward", "flat",   [-0.12,  0.00,  0.00]),
                ("Flat Pure Strafe L",     "flat",   [ 0.00,  0.12,  0.00]),
                ("Flat Pure Strafe R",     "flat",   [ 0.00, -0.12,  0.00]),
                ("Flat In-Place Turn CCW", "flat",   [ 0.00,  0.00,  0.35]),
                ("Flat In-Place Turn CW",  "flat",   [ 0.00,  0.00, -0.35]),
                ("Flat Stand (Planar)",    "flat",   [ 0.00,  0.00,  0.00]),
                ("Rough Forward",          "rough",  [ 0.15,  0.00,  0.00]),
                ("Wavy Forward",           "wavy",   [ 0.15,  0.00,  0.00]),
                ("Slopes Forward",         "slopes", [ 0.15,  0.00,  0.00]),
                ("Mixed Forward",          "mixed",  [ 0.15,  0.00,  0.00]),
            ]

            for test_name, t_type, cmd in locomotion_suite:
                self.eval_env.terrain_type = t_type
                self.eval_env.generate_terrain(mode=t_type, difficulty=0.4)
                obs, _ = self.eval_env.reset(seed=42)
                self.eval_env.set_command(*cmd)
                self.eval_env.target_v = np.array(cmd, dtype=np.float32)
                obs = self.eval_env._get_obs()

                max_tilt = 0.0
                for step in range(300):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, info = self.eval_env.step(action)

                    # Track tilt
                    grav = obs[3:6]
                    tilt = np.degrees(np.arccos(np.clip(-grav[2], -1.0, 1.0)))
                    max_tilt = max(max_tilt, tilt)

                    if "Forward" in test_name or "Trot" in test_name:
                        fwd_speeds.append(info["act_vx"])
                    elif "Backward" in test_name:
                        bwd_speeds.append(info["act_vx"])
                    elif "Strafe" in test_name:
                        strafe_speeds.append(abs(info["act_vy"]))
                    elif "Turn" in test_name:
                        turn_speeds.append(abs(info["act_wz"]))
                        turn_drifts.append(math.sqrt(info["act_vx"]**2 + info["act_vy"]**2))

                    if term or trunc:
                        break

                survived = (step >= 299) and not term
                survivals.append(1.0 if survived else 0.0)
                tilt_maxs.append(max_tilt)

            # Push recovery tests (NEW)
            push_survivals = []
            for push_dir in ["front", "back", "left", "right"]:
                self.eval_env.terrain_type = "flat"
                self.eval_env.generate_terrain(mode="flat", difficulty=0.0)
                obs, _ = self.eval_env.reset(seed=100)
                self.eval_env.set_command(0.0, 0.0, 0.0)
                self.eval_env.target_v = np.zeros(3, dtype=np.float32)
                obs = self.eval_env._get_obs()

                # Let robot settle for 50 steps
                for _ in range(50):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, _, _ = self.eval_env.step(action)
                    if term:
                        break

                if not term:
                    # Apply strong push
                    push_force = {
                        "front": [5.0, 0.0], "back": [-5.0, 0.0],
                        "left": [0.0, 5.0], "right": [0.0, -5.0]
                    }[push_dir]
                    self.eval_env.push_force[:2] = push_force
                    self.eval_env.push_steps = 10
                    self.eval_env.push_active = True

                    # See if robot recovers in 150 steps (3 seconds)
                    for step in range(150):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, term, _, _ = self.eval_env.step(action)
                        if term:
                            break

                push_survived = not term
                push_survivals.append(1.0 if push_survived else 0.0)

            # Calculate scores
            mean_survival = float(np.mean(survivals) * 100.0)
            mean_fwd = float(np.mean(fwd_speeds)) if fwd_speeds else 0.0
            mean_bwd = float(np.mean(bwd_speeds)) if bwd_speeds else 0.0
            mean_strafe = float(np.mean(strafe_speeds)) if strafe_speeds else 0.0
            mean_turn = float(np.mean(turn_speeds)) if turn_speeds else 0.0
            mean_turn_drift = float(np.mean(turn_drifts)) if turn_drifts else 0.0
            mean_tilt = float(np.mean(tilt_maxs))
            push_rate = float(np.mean(push_survivals) * 100.0)

            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / max(elapsed, 1.0))

            # Composite score: survival + push recovery + balanced 4-way locomotion - drift - tilt
            terrain_score = (
                mean_survival * 20.0 +
                push_rate * 15.0 +
                mean_fwd * 1200.0 +
                abs(mean_bwd) * 600.0 +
                mean_strafe * 600.0 +
                mean_turn * 300.0 -
                mean_turn_drift * 500.0 -
                mean_tilt * 5.0
            )

            print("\n" + "=" * 75)
            print(f"[EVAL @ Step {self.num_timesteps:,}] (Elapsed: {elapsed / 60:.1f}m | {fps} FPS)")
            print(f"  * Terrain Survival     : {mean_survival:.1f}% ({int(sum(survivals))}/{len(survivals)})")
            print(f"  * Push Recovery Rate   : {push_rate:.1f}% ({int(sum(push_survivals))}/4)")
            print(f"  * Forward Speed        : {mean_fwd:+.3f} m/s")
            print(f"  * Backward Speed       : {mean_bwd:+.3f} m/s")
            print(f"  * Strafe Speed         : {mean_strafe:.3f} m/s")
            print(f"  * Turn Speed / Drift   : {mean_turn:.3f} rad/s (Drift: {mean_turn_drift:.3f} m/s)")
            print(f"  * Mean Max Tilt        : {mean_tilt:.1f} deg")
            print(f"  * Composite Score      : {terrain_score:.1f}")

            # Get curriculum info
            try:
                envs = self.model.get_env()
                if hasattr(envs, 'envs'):
                    for e in envs.envs:
                        uw = e if hasattr(e, 'curriculum_difficulty') else getattr(e, 'unwrapped', e)
                        if hasattr(uw, 'curriculum_difficulty'):
                            print(f"  * Curriculum Difficulty : {uw.curriculum_difficulty:.3f}")
                            break
            except Exception:
                pass

            if terrain_score > self.best_score:
                self.best_score = terrain_score
                best_path = "models/ppo_barelangS_v2_best.zip"
                self.model.save(best_path)
                print(f"  >>> [NEW BEST] -> {best_path} (Score: {terrain_score:.1f})")
            print("=" * 75 + "\n")
            sys.stdout.flush()

        return True


def make_env_v2(rank, seed=0, difficulty=0.3):
    def _init():
        env = BarelangSTerrainEnvV2(
            training=True,
            terrain_type="random",
            terrain_difficulty=difficulty,
            enable_curriculum=True
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train():
    parser = argparse.ArgumentParser(description="Train BarelangS v2 From Scratch with Push Recovery")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--difficulty", type=float, default=0.2, help="Initial terrain difficulty")
    parser.add_argument("--eval-freq", type=int, default=25000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=200_000, help="Checkpoint save frequency")
    parser.add_argument("--load-model", type=str, default=None, help="Optional: resume from checkpoint")
    parser.add_argument("--best-score", type=float, default=None, help="Initial best score benchmark threshold")
    args = parser.parse_args()

    os.makedirs("checkpoints/v2", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 75)
    print("   BARELANG S v2 - FROM SCRATCH TRAINING WITH PUSH RECOVERY")
    print("=" * 75)
    print(f"Parallel Environments : {args.n_envs}")
    print(f"Total Timesteps       : {args.timesteps:,}")
    print(f"Initial Difficulty    : {args.difficulty}")
    print(f"Evaluation Interval   : Every {args.eval_freq:,} steps")
    print(f"Network Architecture  : MlpPolicy [256, 256]")
    print(f"Curriculum Learning   : ENABLED (auto-scaling difficulty)")
    print(f"Push Recovery Training: ENABLED (progressive force schedule)")
    print("=" * 75)

    if args.n_envs > 1:
        env = SubprocVecEnv([make_env_v2(i, difficulty=args.difficulty) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env_v2(0, difficulty=args.difficulty)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1000),
        save_path="checkpoints/v2/",
        name_prefix="barelangS_v2"
    )
    initial_score = args.best_score if args.best_score is not None else -float("inf")
    eval_callback = TerrainEvalCallbackV2(eval_freq=args.eval_freq, initial_best_score=initial_score)

    if args.load_model and os.path.exists(args.load_model):
        print(f"\nResuming and fine-tuning from checkpoint: {args.load_model}")
        model = PPO.load(args.load_model, env=env)
        model.learning_rate = 1.5e-4
        for pg in model.policy.optimizer.param_groups:
            pg['lr'] = 1.5e-4
        reset_timesteps = False
    else:
        print("\nInitializing FRESH PPO policy (from scratch)...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.008,      # Higher entropy for exploration from scratch
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            tensorboard_log="logs/",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        )
        reset_timesteps = True

    print("Starting extended training with push recovery and precision locomotion...\n")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            reset_num_timesteps=reset_timesteps,
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")

    final_path = "models/ppo_barelangS_v2_final.zip"
    model.save(final_path)
    print(f"\nTraining completed! Final model: {final_path}")


if __name__ == "__main__":
    train()
