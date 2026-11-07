import os
import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from barelang_terrain_env import BarelangSTerrainEnv

class TerrainEvaluationCallback(BaseCallback):
    """
    Periodically evaluates policy across diverse 3D terrains (rough, wavy, slopes, mixed)
    and all 6 motion commands, saving the best terrain-adapted locomotion model.
    """
    def __init__(self, eval_freq=25000, n_eval_episodes=2, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_score = -float("inf")
        self.eval_env = None
        self.start_time = None

    def _on_training_start(self) -> None:
        self.eval_env = BarelangSTerrainEnv(training=False, terrain_type="random", terrain_difficulty=0.7)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.model.n_envs:
            survivals = []
            rewards = []
            tracking_errors = []
            fwd_speeds = []
            bwd_speeds = []

            eval_suite = [
                ("Rough Forward",   "rough",  [ 0.20,  0.00,  0.00]),
                ("Wavy Forward",    "wavy",   [ 0.20,  0.00,  0.00]),
                ("Slopes Forward",  "slopes", [ 0.20,  0.00,  0.00]),
                ("Mixed Backward",  "mixed",  [-0.12,  0.00,  0.00]),
                ("Rough Strafe L",  "rough",  [ 0.00,  0.12,  0.00]),
                ("Wavy Strafe R",   "wavy",   [ 0.00, -0.12,  0.00]),
                ("Slopes Turn CCW", "slopes", [ 0.00,  0.00,  0.35]),
                ("Rough Stand",     "rough",  [ 0.00,  0.00,  0.00]),
            ]

            for test_name, t_type, cmd in eval_suite:
                self.eval_env.terrain_type = t_type
                for ep in range(self.n_eval_episodes):
                    self.eval_env.generate_terrain(mode=t_type, difficulty=self.eval_env.terrain_difficulty)
                    obs, _ = self.eval_env.reset(seed=100 + ep)
                    self.eval_env.set_command(*cmd)
                    self.eval_env.target_v = np.array(cmd, dtype=np.float32)
                    obs = self.eval_env._get_obs()
                    ep_reward = 0.0
                    vel_errors = []

                    for step in range(400): # 8.0 seconds on terrain
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, term, trunc, info = self.eval_env.step(action)
                        ep_reward += reward

                        act_vel = np.array([info["act_vx"], info["act_vy"], info["act_wz"]])
                        target_vel = np.array(cmd)
                        vel_errors.append(np.linalg.norm(act_vel - target_vel))

                        if "Forward" in test_name:
                            fwd_speeds.append(info["act_vx"])
                        elif "Backward" in test_name:
                            bwd_speeds.append(info["act_vx"])

                        if term or trunc:
                            break

                    survived = (step >= 399) and not term
                    survivals.append(1.0 if survived else 0.0)
                    rewards.append(ep_reward)
                    tracking_errors.append(np.mean(vel_errors))

            mean_survival = float(np.mean(survivals) * 100.0)
            mean_reward = float(np.mean(rewards))
            mean_error = float(np.mean(tracking_errors))
            mean_fwd_speed = float(np.mean(fwd_speeds)) if fwd_speeds else 0.0
            mean_bwd_speed = float(np.mean(bwd_speeds)) if bwd_speeds else 0.0
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / max(elapsed, 1.0))

            terrain_score = (mean_survival * 25.0) + (mean_fwd_speed * 1800.0) - (mean_error * 500.0) + (mean_reward * 0.05)

            print("\n" + "=" * 75)
            print(f"[TERRAIN EVALUATION @ Step {self.num_timesteps:,}] (Elapsed: {elapsed/60:.1f}m | {fps} FPS)")
            print(f"  * Terrain Survival Rate   : {mean_survival:.1f} % (Rough, Wavy, Slopes, Mixed)")
            print(f"  * Forward Speed on Terrain: {mean_fwd_speed:+.3f} m/s")
            print(f"  * Backward Speed on Terrain:{mean_bwd_speed:+.3f} m/s")
            print(f"  * Tracking Error on Terrain:{mean_error:.4f}")
            print(f"  * Mean Terrain Reward     : {mean_reward:.2f}")
            print(f"  * Terrain Score           : {terrain_score:.1f}")

            if terrain_score > self.best_score:
                self.best_score = terrain_score
                best_path = "models/ppo_barelangS_terrain_best.zip"
                self.model.save(best_path)
                print(f"  >>> [NEW BEST TERRAIN MODEL SAVED] -> {best_path} (Survival: {mean_survival:.1f}%)")
            print("=" * 75 + "\n")

        return True

def make_terrain_env(rank, seed=0, difficulty=0.6):
    def _init():
        env = BarelangSTerrainEnv(
            training=True, 
            terrain_type="random", 
            terrain_difficulty=difficulty
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    parser = argparse.ArgumentParser(description="Train BarelangS Humanoid on Procedural 3D Terrain")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total training timesteps (default: 1.5M)")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--load-model", type=str, default="models/ppo_barelangS_best.zip", help="Path to checkpoint model")
    parser.add_argument("--difficulty", type=float, default=0.6, help="Terrain bump/slope difficulty [0.1 - 1.0]")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint frequency in steps")
    parser.add_argument("--eval-freq", type=int, default=30_000, help="Evaluation frequency in steps")
    args = parser.parse_args()

    os.makedirs("checkpoints/terrain", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 75)
    print("      BARELANG S PROCEDURAL TERRAIN ADAPTIVE LOCOMOTION TRAINING")
    print("=" * 75)
    print(f"Parallel Environments : {args.n_envs}")
    print(f"Total Timesteps       : {args.timesteps:,}")
    print(f"Initial Checkpoint    : {args.load_model}")
    print(f"Terrain Difficulty    : {args.difficulty} (Bumps, Rolling Waves, Slopes)")
    print(f"Evaluation Interval   : Every {args.eval_freq:,} steps")
    print("=" * 75)

    if args.n_envs > 1:
        env = SubprocVecEnv([make_terrain_env(i, difficulty=args.difficulty) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_terrain_env(0, difficulty=args.difficulty)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1000),
        save_path="checkpoints/terrain/",
        name_prefix="barelangS_terrain"
    )
    eval_callback = TerrainEvaluationCallback(eval_freq=args.eval_freq, n_eval_episodes=2)

    if args.load_model and os.path.exists(args.load_model):
        print(f"Resuming / Transferring policy from: {args.load_model}...")
        model = PPO.load(args.load_model, env=env)
    else:
        print("Initializing new PPO policy network...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
            tensorboard_log="logs/"
        )

    print("Starting procedural terrain adaptive training...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current terrain policy...")

    final_model_path = "models/ppo_barelangS_terrain_final.zip"
    model.save(final_model_path)
    print(f"Terrain training completed! Final model saved to: {final_model_path}")

if __name__ == "__main__":
    train()
