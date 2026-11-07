import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from barelang_env import BarelangSWalkEnv

# GLFW Keycodes
KEY_W = 87
KEY_S = 83
KEY_A = 65
KEY_D = 68
KEY_Q = 81
KEY_E = 69
KEY_SPACE = 32
KEY_R = 82
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262

class Sim2SimKeyboardController:
    """
    Sim2Sim Interactive Keyboard Controller for BarelangS Humanoid Walking.
    Features:
      - Slew-rate velocity ramping (natural acceleration without tipping/falling).
      - Passive MuJoCo camera tracking following the robot.
      - Real-time terminal HUD overlay.
    """
    def __init__(self, model_path=None):
        self.env = BarelangSWalkEnv(training=False)
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.reset_requested = False
        
        # Load trained policy if available
        self.model = None
        if model_path is None:
            candidates = [
                "models/ppo_barelangS_best.zip",
                "models/ppo_barelangS_walk.zip"
            ]
            if os.path.exists("checkpoints"):
                cps = [os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".zip")]
                cps.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                candidates.extend(cps)
                
            for cand in candidates:
                if os.path.exists(cand):
                    model_path = cand
                    break

        if model_path and os.path.exists(model_path):
            print(f"Loading trained walking policy: {model_path}...")
            self.model = PPO.load(model_path)
            print("Policy loaded successfully!")
        else:
            print("Notice: No trained policy checkpoint found. Running in baseline CPG walking mode.")

    def key_callback(self, keycode: int):
        """Callback for keyboard events from MuJoCo viewer window."""
        # Forward / Backward
        if keycode in (KEY_W, KEY_UP):
            self.cmd_vx = round(min(self.cmd_vx + 0.05, 0.25), 2)
        elif keycode in (KEY_S, KEY_DOWN):
            self.cmd_vx = round(max(self.cmd_vx - 0.05, -0.15), 2)
            
        # Lateral Left / Right
        elif keycode in (KEY_A, KEY_LEFT):
            self.cmd_vy = round(min(self.cmd_vy + 0.05, 0.16), 2)
        elif keycode in (KEY_D, KEY_RIGHT):
            self.cmd_vy = round(max(self.cmd_vy - 0.05, -0.16), 2)
            
        # Rotation Yaw (Q: Counter-Clockwise, E: Clockwise)
        elif keycode == KEY_Q:
            self.cmd_wz = round(min(self.cmd_wz + 0.08, 0.45), 2)
        elif keycode == KEY_E:
            self.cmd_wz = round(max(self.cmd_wz - 0.08, -0.45), 2)
            
        # Space: Emergency Stop / Stand In Place
        elif keycode == KEY_SPACE:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0
            
        # R: Reset robot pose
        elif keycode == KEY_R:
            self.reset_requested = True

    def run(self):
        print("=" * 70)
        print("     BARELANG S SIM2SIM INTERACTIVE KEYBOARD CONTROLLER")
        print("=" * 70)
        print("Controls (Click inside the MuJoCo viewer window):")
        print("  [W] or [Up Arrow]    : Walk Forward               (+0.05 m/s, max: +0.25 m/s)")
        print("  [S] or [Down Arrow]  : Walk Backward              (-0.05 m/s, min: -0.15 m/s)")
        print("  [A] or [Left Arrow]  : Strafe Left                (+0.05 m/s, max: +0.16 m/s)")
        print("  [D] or [Right Arrow] : Strafe Right               (-0.05 m/s, min: -0.16 m/s)")
        print("  [Q]                  : Turn Left (Yaw CCW)        (+0.08 rad/s, max: +0.45 rad/s)")
        print("  [E]                  : Turn Right (Yaw CW)        (-0.08 rad/s, min: -0.45 rad/s)")
        print("  [SPACE]              : Stop / Stand in Place")
        print("  [R]                  : Reset Robot to Origin")
        print("=" * 70)

        obs, _ = self.env.reset()
        last_hud_time = 0.0

        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=self.key_callback) as viewer:
            torso_bid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
            if torso_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = torso_bid
                viewer.cam.distance = 2.2
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

                # Set user velocity commands
                self.env.set_command(self.cmd_vx, self.cmd_vy, self.cmd_wz)

                # Predict residual action
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = np.zeros(self.env.num_leg_joints, dtype=np.float32)

                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    print(f"\n[ALERT] Fall detected (Torso Z: {info['torso_z']:.3f}m)! Auto-recovering...")
                    obs, _ = self.env.reset()
                    self.cmd_vx = 0.0
                    self.cmd_vy = 0.0
                    self.cmd_wz = 0.0

                viewer.sync()

                # Print terminal status dashboard every 0.15s
                now = time.time()
                if now - last_hud_time > 0.15:
                    last_hud_time = now
                    sys.stdout.write(
                        f"\r[CMD] Vx: {self.cmd_vx:+5.2f} m/s | Vy: {self.cmd_vy:+5.2f} m/s | Wz: {self.cmd_wz:+5.2f} rad/s "
                        f"|| [ACTUAL] Vx: {info['act_vx']:+5.2f} m/s | Vy: {info['act_vy']:+5.2f} m/s | Torso Z: {info['torso_z']:.3f}m  "
                    )
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < self.env.dt:
                    time.sleep(self.env.dt - elapsed)

        print("\nViewer closed. Sim2Sim session ended.")

def main():
    parser = argparse.ArgumentParser(description="Sim2Sim Keyboard Controller for BarelangS")
    parser.add_argument("--model", type=str, default=None, help="Path to trained policy .zip")
    args = parser.parse_args()

    controller = Sim2SimKeyboardController(model_path=args.model)
    controller.run()

if __name__ == "__main__":
    main()
