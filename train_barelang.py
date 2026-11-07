import os
import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from barelang_env import BarelangSWalkEnv

class AutoEvaluationCallback(BaseCallback):
    """
    Periodically evaluates the policy across diverse commands,
    computes survival rate, actual forward speed, and saves the best model.
    """
    def __init__(self, eval_freq=25000, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_score = -float("inf")
        self.eval_env = None
        self.start_time = None

    def _on_training_start(self) -> None:
        self.eval_env = BarelangSWalkEnv(training=False)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Check every eval_freq total steps
        if self.num_timesteps % self.eval_freq < self.model.n_envs:
            survivals = []
            rewards = []
            tracking_errors = []
            fwd_speeds = []

            bwd_speeds = []

            eval_commands = [
                ("Forward",  [ 0.25,  0.00,  0.00]),
                ("Backward", [-0.15,  0.00,  0.00]),
                ("Strafe L", [ 0.00,  0.12,  0.00]),
                ("Strafe R", [ 0.00, -0.12,  0.00]),
                ("Turn CCW", [ 0.00,  0.00,  0.35]),
                ("Turn CW",  [ 0.00,  0.00, -0.35]),
            ]

            for mode_name, cmd in eval_commands:
                for ep in range(self.n_eval_episodes):
                    obs, _ = self.eval_env.reset()
                    self.eval_env.set_command(*cmd)
                    ep_reward = 0.0
                    vel_errors = []
                    vx_history = []
                    bwd_history = []

                    for step in range(400):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, term, trunc, info = self.eval_env.step(action)
                        ep_reward += reward
                        
                        act_vel = np.array([info["act_vx"], info["act_vy"], info["act_wz"]])
                        target_vel = np.array(cmd)
                        vel_errors.append(np.linalg.norm(act_vel - target_vel))
                        if mode_name == "Forward":
                            vx_history.append(info["act_vx"])
                        elif mode_name == "Backward":
                            bwd_history.append(info["act_vx"])
                        
                        if term or trunc:
                            break

                    survived = (step >= 399) and not term
                    survivals.append(1.0 if survived else 0.0)
                    rewards.append(ep_reward)
                    tracking_errors.append(np.mean(vel_errors))
                    if vx_history:
                        fwd_speeds.append(np.mean(vx_history))
                    if bwd_history:
                        bwd_speeds.append(np.mean(bwd_history))

            mean_survival = float(np.mean(survivals) * 100.0)
            mean_reward = float(np.mean(rewards))
            mean_error = float(np.mean(tracking_errors))
            mean_fwd_speed = float(np.mean(fwd_speeds)) if fwd_speeds else 0.0
            mean_bwd_speed = float(np.mean(bwd_speeds)) if bwd_speeds else 0.0
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / max(elapsed, 1.0))

            # Omnidirectional Walking Score prioritizes 100% survival across all modes and low error
            walking_score = (mean_survival * 25.0) - (mean_error * 800.0) + (mean_reward * 0.05)

            print("\n" + "=" * 70)
            print(f"[EVALUATION @ Step {self.num_timesteps:,}] (Elapsed: {elapsed/60:.1f}m | {fps} FPS)")
            print(f"  * Total Survival Rate     : {mean_survival:.1f} % (Across All 6 Modes)")
            print(f"  * Forward Speed           : {mean_fwd_speed:+.3f} m/s")
            print(f"  * Backward Speed          : {mean_bwd_speed:+.3f} m/s")
            print(f"  * Velocity Tracking Error : {mean_error:.4f}")
            print(f"  * Mean Episode Reward     : {mean_reward:.2f}")
            print(f"  * Omnidirectional Score   : {walking_score:.1f}")
            
            # Save if new best omnidirectional walking performance
            if walking_score > self.best_score:
                self.best_score = walking_score
                best_path = "models/ppo_barelangS_best.zip"
                self.model.save(best_path)
                print(f"  >>> [NEW BEST OMNIDIRECTIONAL MODEL SAVED] -> {best_path} (Survival: {mean_survival:.1f}%)")
            print("=" * 70 + "\n")

        return True

def make_env(rank, seed=0):
    def _init():
        env = BarelangSWalkEnv(training=True)
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    parser = argparse.ArgumentParser(description="Train BarelangS Humanoid Walking with Active Stepping Rewards")
    parser.add_argument("--timesteps", type=int, default=18_000_000, help="Total training timesteps (default: 18M for overnight)")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--load-model", type=str, default="models/ppo_barelangS_best.zip", help="Path to checkpoint model to resume from")
    parser.add_argument("--save-freq", type=int, default=250_000, help="Checkpoint frequency in steps")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency in steps")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 70)
    print("      BARELANG S ACTIVE STEPPING LOCOMOTION TRAINING (PPO)")
    print("=" * 70)
    print(f"Parallel Environments : {args.n_envs}")
    print(f"Total Timesteps       : {args.timesteps:,}")
    print(f"Actuator Spec         : MX-106 (Legs), MX-64 (Hip Yaw & Arms), MX-28 (Head)")
    print(f"Gait Controller       : Active Cubic-Bezier CPG + Bit-Bots Touchdown Rewards")
    print(f"Control Frequency     : 50 Hz (20ms step)")
    print(f"Evaluation Interval   : Every {args.eval_freq:,} steps")
    print("=" * 70)

    # Initialize vectorized environment across CPU cores
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1000),
        save_path="checkpoints/",
        name_prefix="barelangS_ppo"
    )
    eval_callback = AutoEvaluationCallback(eval_freq=args.eval_freq, n_eval_episodes=3)

    if args.load_model and os.path.exists(args.load_model):
        print(f"Resuming training from checkpoint: {args.load_model}...")
        model = PPO.load(args.load_model, env=env)
    else:
        print("Initializing new PPO policy network (Actor-Critic MLP [256, 256])...")
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

    print("Starting active stepping training pipeline...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current policy state...")

    final_model_path = "models/ppo_barelangS_walk.zip"
    model.save(final_model_path)
    print("=" * 70)
    print(f"Training completed / saved successfully to: {final_model_path}")
    print("=" * 70)

if __name__ == "__main__":
    train()
