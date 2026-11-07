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
KEY_I = 73
KEY_K = 75
KEY_J = 74
KEY_L = 76
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262

class Sim2SimTerrainKeyboardController:
    """
    Interactive 3D Keyboard Controller for Barelang S on Procedural 3D Terrain.
    Supports real-time terrain switching, elevation tracking, and velocity controls.
    """
    def __init__(self, model_path=None, terrain_type="rough", difficulty=0.6):
        self.terrain_type = terrain_type
        self.difficulty = difficulty
        self.env = BarelangSTerrainEnvV2(
            training=False, 
            terrain_type=self.terrain_type, 
            terrain_difficulty=self.difficulty
        )

        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading terrain policy: {model_path}...")
            self.model = PPO.load(model_path)
            print("Policy loaded successfully!")
        elif os.path.exists("models/ppo_barelangS_v2_best.zip"):
            print("Loading default v2 best model: models/ppo_barelangS_v2_best.zip...")
            self.model = PPO.load("models/ppo_barelangS_v2_best.zip")
        elif os.path.exists("models/ppo_barelangS_terrain_best.zip"):
            print("Loading legacy terrain model: models/ppo_barelangS_terrain_best.zip...")
            self.model = PPO.load("models/ppo_barelangS_terrain_best.zip")
        elif os.path.exists("models/ppo_barelangS_best.zip"):
            print("Loading best flat policy: models/ppo_barelangS_best.zip...")
            self.model = PPO.load("models/ppo_barelangS_best.zip")
        else:
            print("Warning: No checkpoint found, running in pure CPG baseline mode.")

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.reset_requested = False
        self.terrain_modes = ["rough", "wavy", "slopes", "mixed", "flat"]
        self.terrain_descriptions = {
            "rough": "Rocky Gravel & Broken Dirt",
            "wavy": "Alpine Meadow Grass & Earth",
            "slopes": "Canyon Sandstone & Ridge Slope",
            "mixed": "Highland Mountain Trail",
            "flat": "Outdoor Testing Ground"
        }
        self.terrain_idx = self.terrain_modes.index(self.terrain_type) if self.terrain_type in self.terrain_modes else 0

    def key_callback(self, keycode):
        # W / Up: Forward
        if keycode in [KEY_W, KEY_UP]:
            self.cmd_vx = round(min(self.cmd_vx + 0.05, 0.25), 2)
        # S / Down: Backward
        elif keycode in [KEY_S, KEY_DOWN]:
            self.cmd_vx = round(max(self.cmd_vx - 0.05, -0.15), 2)
        # A / Left: Strafe Left
        elif keycode in [KEY_A, KEY_LEFT]:
            self.cmd_vy = round(min(self.cmd_vy + 0.05, 0.16), 2)
        # D / Right: Strafe Right
        elif keycode in [KEY_D, KEY_RIGHT]:
            self.cmd_vy = round(max(self.cmd_vy - 0.05, -0.16), 2)
        # Q: Turn Left (CCW)
        elif keycode == KEY_Q:
            self.cmd_wz = round(min(self.cmd_wz + 0.08, 0.45), 2)
        # E: Turn Right (CW)
        elif keycode == KEY_E:
            self.cmd_wz = round(max(self.cmd_wz - 0.08, -0.45), 2)
        # T: Cycle Terrain Type with continuous elevation under current position
        elif keycode == KEY_T:
            self.terrain_idx = (self.terrain_idx + 1) % len(self.terrain_modes)
            self.terrain_type = self.terrain_modes[self.terrain_idx]
            cur_pos = self.env.data.qpos[:2]
            self.env.generate_terrain(mode=self.terrain_type, difficulty=self.difficulty, center_x=cur_pos[0], center_y=cur_pos[1])
            desc = self.terrain_descriptions.get(self.terrain_type, "")
            print(f"\n[TERRAIN CHANGED] Active Mode: -> {self.terrain_type.upper()} ({desc})")
        # Space: Stop / Stand
        elif keycode == KEY_SPACE:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0
        # R: Reset robot
        elif keycode == KEY_R:
            self.reset_requested = True
        # Push Disturbance testing (Torso IMU Balance Response)
        elif keycode == KEY_I:
            self.env.push_force[:2] = [8.0, 0.0]
            self.env.push_steps = 10
            print("\n[PUSH DISTURBANCE] Applied +8.0 N FRONT push (IMU torso correction active)")
        elif keycode == KEY_K:
            self.env.push_force[:2] = [-8.0, 0.0]
            self.env.push_steps = 10
            print("\n[PUSH DISTURBANCE] Applied -8.0 N BACK push (IMU torso correction active)")
        elif keycode == KEY_J:
            self.env.push_force[:2] = [0.0, 8.0]
            self.env.push_steps = 10
            print("\n[PUSH DISTURBANCE] Applied +8.0 N LEFT push (IMU torso correction active)")
        elif keycode == KEY_L:
            self.env.push_force[:2] = [0.0, -8.0]
            self.env.push_steps = 10
            print("\n[PUSH DISTURBANCE] Applied -8.0 N RIGHT push (IMU torso correction active)")

    def run(self):
        print("=" * 75)
        print("     BARELANG S PROCEDURAL TERRAIN SIM2SIM KEYBOARD VIEWER")
        print("=" * 75)
        print(f"Active Terrain Mode   : {self.terrain_type.upper()} (Difficulty: {self.difficulty})")
        print("Controls (Click inside the MuJoCo viewer window):")
        print("  [W] or [Up Arrow]    : Walk Forward               (+0.05 m/s, max: +0.25 m/s)")
        print("  [S] or [Down Arrow]  : Walk Backward              (-0.05 m/s, min: -0.15 m/s)")
        print("  [A] or [Left Arrow]  : Strafe Left                (+0.05 m/s, max: +0.16 m/s)")
        print("  [D] or [Right Arrow] : Strafe Right               (-0.05 m/s, min: -0.16 m/s)")
        print("  [Q]                  : Turn Left (Yaw CCW)        (+0.08 rad/s, max: +0.45 rad/s)")
        print("  [E]                  : Turn Right (Yaw CW)        (-0.08 rad/s, min: -0.45 rad/s)")
        print("  [T]                  : Switch Terrain Type        (Rough -> Wavy -> Slopes -> Mixed -> Flat)")
        print("  [SPACE]              : Stop / Stand in Place      (Planar Flat Foot Soles)")
        print("  [R]                  : Reset Robot to Origin")
        print("  --- UJI DORONGAN & KOREKSI IMU TORSO (PUSH DISTURBANCE) ---")
        print("  [I]                  : Dorong Torso Depan         (+8.0 N)")
        print("  [K]                  : Dorong Torso Belakang      (-8.0 N)")
        print("  [J]                  : Dorong Torso Samping Kiri  (+8.0 N)")
        print("  [L]                  : Dorong Torso Samping Kanan (-8.0 N)")
        print("=" * 75)

        obs, _ = self.env.reset()
        last_hud_time = 0.0

        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=self.key_callback) as viewer:
            torso_bid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
            if torso_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = torso_bid
                viewer.cam.distance = 2.4
                viewer.cam.elevation = -18.0
                viewer.cam.azimuth = 135.0

            print("\nViewer launched! Camera tracking active. Ready for keyboard input.")

            while viewer.is_running():
                step_start = time.time()

                if self.reset_requested:
                    obs, _ = self.env.reset()
                    self.cmd_vx, self.cmd_vy, self.cmd_wz = 0.0, 0.0, 0.0
                    self.reset_requested = False
                    print("\n[RESET] Robot re-initialized to default standing pose.")

                self.env.set_command(self.cmd_vx, self.cmd_vy, self.cmd_wz)

                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(self.env.num_leg_joints, dtype=np.float32)

                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    print(f"\n[ALERT] Fall detected on terrain! Auto-recovering...")
                    obs, _ = self.env.reset()
                    self.cmd_vx, self.cmd_vy, self.cmd_wz = 0.0, 0.0, 0.0

                viewer.sync()

                now = time.time()
                if now - last_hud_time > 0.15:
                    last_hud_time = now
                    sys.stdout.write(
                        f"\r[TERRAIN: {self.terrain_type.upper():<6}] [CMD] Vx: {self.cmd_vx:+5.2f} Vy: {self.cmd_vy:+5.2f} Wz: {self.cmd_wz:+5.2f} "
                        f"|| [ACT] Vx: {info['act_vx']:+5.2f} Vy: {info['act_vy']:+5.2f} || TorsoZ: {info['torso_z']:.3f}m GroundZ: {info['terrain_z']:.3f}m  "
                    )
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < self.env.dt:
                    time.sleep(self.env.dt - elapsed)

        print("\nViewer closed. Terrain Sim2Sim session ended.")

def main():
    parser = argparse.ArgumentParser(description="Sim2Sim Procedural Terrain Keyboard Controller for BarelangS")
    parser.add_argument("--model", type=str, default=None, help="Path to trained policy .zip")
    parser.add_argument("--terrain", type=str, default="rough", choices=["rough", "wavy", "slopes", "mixed", "flat"], help="Initial terrain type")
    parser.add_argument("--difficulty", type=float, default=0.6, help="Terrain bump/slope difficulty [0.1 - 1.0]")
    args = parser.parse_args()

    controller = Sim2SimTerrainKeyboardController(
        model_path=args.model, 
        terrain_type=args.terrain, 
        difficulty=args.difficulty
    )
    controller.run()

if __name__ == "__main__":
    main()
