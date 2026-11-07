"""
Sim2Sim Interactive Viewer for Barelang S v2
=============================================
Supports push testing with [P] key and all terrain modes.
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from barelang_terrain_env_v2 import BarelangSTerrainEnvV2

# GLFW Key Constants
KEY_SPACE = 32
KEY_W = 87
KEY_S = 83
KEY_A = 65
KEY_D = 68
KEY_Q = 81
KEY_E = 69
KEY_R = 82
KEY_T = 84
KEY_P = 80
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262


class Sim2SimV2Controller:
    """Interactive controller with push testing support."""

    def __init__(self, model_path=None, terrain_type="flat", difficulty=0.3):
        self.terrain_type = terrain_type
        self.difficulty = difficulty
        self.env = BarelangSTerrainEnvV2(
            training=False,
            terrain_type=self.terrain_type,
            terrain_difficulty=self.difficulty,
            enable_curriculum=False
        )
        self.env.generate_terrain(mode=self.terrain_type, difficulty=self.difficulty)

        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading v2 policy: {model_path}...")
            self.model = PPO.load(model_path)
            print("Policy loaded!")
        elif os.path.exists("models/ppo_barelangS_v2_best.zip"):
            print("Loading default v2 best model...")
            self.model = PPO.load("models/ppo_barelangS_v2_best.zip")
        else:
            print("Warning: No model found, running pure CPG baseline.")

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.reset_requested = False
        self.terrain_modes = ["rough", "wavy", "slopes", "mixed", "flat"]
        self.terrain_idx = self.terrain_modes.index(terrain_type) if terrain_type in self.terrain_modes else 4

    def key_callback(self, keycode):
        if keycode in [KEY_W, KEY_UP]:
            self.cmd_vx = round(min(self.cmd_vx + 0.05, 0.22), 2)
        elif keycode in [KEY_S, KEY_DOWN]:
            self.cmd_vx = round(max(self.cmd_vx - 0.05, -0.15), 2)
        elif keycode in [KEY_A, KEY_LEFT]:
            self.cmd_vy = round(min(self.cmd_vy + 0.05, 0.15), 2)
        elif keycode in [KEY_D, KEY_RIGHT]:
            self.cmd_vy = round(max(self.cmd_vy - 0.05, -0.15), 2)
        elif keycode == KEY_Q:
            self.cmd_wz = round(min(self.cmd_wz + 0.08, 0.40), 2)
        elif keycode == KEY_E:
            self.cmd_wz = round(max(self.cmd_wz - 0.08, -0.40), 2)
        elif keycode == KEY_T:
            self.terrain_idx = (self.terrain_idx + 1) % len(self.terrain_modes)
            self.terrain_type = self.terrain_modes[self.terrain_idx]
            cur_pos = self.env.data.qpos[:2]
            self.env.generate_terrain(mode=self.terrain_type, difficulty=self.difficulty,
                                      center_x=cur_pos[0], center_y=cur_pos[1])
            print(f"\n[TERRAIN] -> {self.terrain_type.upper()}")
        elif keycode == KEY_P:
            # Apply a random strong push to test recovery
            angle = np.random.uniform(0, 2 * math.pi)
            mag = 6.0
            self.env.push_force[0] = mag * math.cos(angle)
            self.env.push_force[1] = mag * math.sin(angle)
            self.env.push_steps = 10
            self.env.push_active = True
            dir_name = {0: "Front", 1: "Right", 2: "Back", 3: "Left"}[int((angle / (math.pi / 2)) % 4)]
            print(f"\n[PUSH!] Direction: ~{dir_name} | Force: {mag:.1f}N x 10 steps")
        elif keycode == KEY_SPACE:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0
        elif keycode == KEY_R:
            self.reset_requested = True

    def run(self):
        print("=" * 75)
        print("     BARELANG S v2 - SIM2SIM VIEWER WITH PUSH RECOVERY")
        print("=" * 75)
        print(f"Terrain: {self.terrain_type.upper()} (Difficulty: {self.difficulty})")
        print("Controls:")
        print("  [W/S]  Forward/Backward    [A/D] Strafe L/R")
        print("  [Q/E]  Turn CCW/CW         [T]   Switch Terrain")
        print("  [P]    PUSH TEST!           [SPACE] Stop")
        print("  [R]    Reset")
        print("=" * 75)

        obs, _ = self.env.reset()
        last_hud = 0.0

        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=self.key_callback) as viewer:
            torso_bid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
            if torso_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = torso_bid
                viewer.cam.distance = 2.4
                viewer.cam.elevation = -18.0
                viewer.cam.azimuth = 135.0

            print("\nViewer ready!")

            while viewer.is_running():
                step_start = time.time()

                if self.reset_requested:
                    obs, _ = self.env.reset()
                    self.cmd_vx, self.cmd_vy, self.cmd_wz = 0.0, 0.0, 0.0
                    self.reset_requested = False
                    print("\n[RESET]")

                self.env.set_command(self.cmd_vx, self.cmd_vy, self.cmd_wz)

                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(self.env.num_leg_joints, dtype=np.float32)

                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    print(f"\n[FALL] Auto-recovering...")
                    obs, _ = self.env.reset()
                    self.cmd_vx, self.cmd_vy, self.cmd_wz = 0.0, 0.0, 0.0

                viewer.sync()

                now = time.time()
                if now - last_hud > 0.15:
                    last_hud = now
                    push_str = " [PUSH!]" if info.get("push_active", False) else ""
                    sys.stdout.write(
                        f"\r[{self.terrain_type.upper():<6}] "
                        f"Cmd Vx:{self.cmd_vx:+5.2f} Vy:{self.cmd_vy:+5.2f} Wz:{self.cmd_wz:+5.2f} | "
                        f"Act Vx:{info['act_vx']:+5.2f} Vy:{info['act_vy']:+5.2f} | "
                        f"H:{info['rel_torso_z']:.3f}m{push_str}  "
                    )
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < self.env.dt:
                    time.sleep(self.env.dt - elapsed)

        print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim v2 Viewer")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--terrain", type=str, default="flat",
                        choices=["rough", "wavy", "slopes", "mixed", "flat"])
    parser.add_argument("--difficulty", type=float, default=0.3)
    args = parser.parse_args()

    controller = Sim2SimV2Controller(
        model_path=args.model,
        terrain_type=args.terrain,
        difficulty=args.difficulty
    )
    controller.run()


if __name__ == "__main__":
    main()
