#!/usr/bin/env python3
"""
steer_barelang_supermimic.py
----------------------------
Interactive Omnidirectional Steering Controller for BarelangS Humanoid
Powered by SuperMimic and Bio-mechanical Human Motion Capture.

Keyboard Controls (In MuJoCo Viewer):
  [W]          : Walk Forward (increase speed)
  [S]          : Slow Down / Walk Backward / Stop
  [A]          : Steer / Turn Left
  [D]          : Steer / Turn Right
  [Q]          : Strafe Left
  [E]          : Strafe Right
  [SPACE]      : Instant Stop / Stand Still
  [R]          : Reset to Origin (0, 0)
  [1] / [2]    : Decrease / Increase Speed Preset
  [V]          : Toggle Camera Mode (Chase Cam vs Orbit Cam)
  [ESC]        : Exit

Usage:
  # Interactive 3D Window with WASD controls:
  python steer_barelang_supermimic.py

  # Headless video demonstration of steering path (S-curve & turning):
  python steer_barelang_supermimic.py --render-demo data/mocap/barelang_steering_demo.mp4
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import scipy.spatial.transform as sptr
import mujoco

# GLFW Key Constants
KEY_SPACE = 32
KEY_W = 87
KEY_S = 83
KEY_A = 65
KEY_D = 68
KEY_Q = 81
KEY_E = 69
KEY_R = 82
KEY_V = 86
KEY_1 = 49
KEY_2 = 50

# Canonical Leg Indices
# [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll,
#  right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll]
LEG_QPOS_ADR = [21, 22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20]
LEG_ACT_ID   = [12, 0, 1, 2, 3, 4, 13, 5, 6, 7, 8, 9]

class SuperMimicSteeringController:
    def __init__(
        self,
        mocap_path: str = "data/mocap/barelang_mocap_walk.npz",
        xml_path: str = "barelangS_terrain_mujoco.xml",
    ):
        print(f"Loading MoCap gait library: {mocap_path}")
        mocap = np.load(mocap_path)
        self.raw_qpos = mocap["qpos"].astype(np.float32)
        self.raw_qvel = mocap["qvel"].astype(np.float32)
        self.fps = int(mocap["fps"])
        self.dt = float(mocap["dt"])
        self.n_frames = len(self.raw_qpos)
        self.cycle_start = int(mocap.get("cycle_start", 18))
        self.cycle_end = int(mocap.get("cycle_end", 80))
        self.cycle_len = self.cycle_end - self.cycle_start
        self.nominal_speed = float(mocap.get("walking_speed", 0.710))

        print(f"Loading MuJoCo physics model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Compliant Standing Pose (qpos for 12 legs in canonical order)
        # Left leg:  hip_yaw=0, hip_roll=0, hip_pitch=-0.20, knee=0.30, ankle_p=-0.10, ankle_r=0
        # Right leg: hip_yaw=0, hip_roll=0, hip_pitch=-0.20, knee=0.30, ankle_p=-0.10, ankle_r=0
        self.standing_legs = np.array([
            0.0, 0.0, -0.20, 0.30, -0.10, 0.0,  # Left
            0.0, 0.0, -0.20, 0.30, -0.10, 0.0   # Right
        ], dtype=np.float32)
        self.standing_root_z = -0.0045

        # Upper body posture
        self.upper_posture = {
            "right_shoulder_pitch": 1.57,
            "left_shoulder_pitch": 1.57,
            "right_shoulder_roll": -0.15,
            "left_shoulder_roll": 0.15,
            "right_elbow": -0.80,
            "left_elbow": -0.80,
            "head_yaw": 0.0,
            "head_pitch": -0.30
        }
        self.upper_act_dict = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, k): v
            for k, v in self.upper_posture.items()
        }

        # Command velocities
        self.cmd_vx = 0.0      # Forward speed (m/s) [-0.3, 1.0]
        self.cmd_vy = 0.0      # Lateral speed (m/s) [-0.3, 0.3]
        self.cmd_wz = 0.0      # Yaw rate (rad/s)    [-1.2, 1.2]
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0

        # Robot state
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.heading = 0.0     # Yaw in radians
        self.gait_phase = float(self.cycle_start)  # Floating frame index
        self.stand_blend = 1.0 # 1.0 = full stand, 0.0 = full walk
        self.chase_cam = True

        self.reset_to_origin()

    def reset_to_origin(self):
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.heading = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0
        self.gait_phase = float(self.cycle_start)
        self.stand_blend = 1.0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.standing_root_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0

        for i, qadr in enumerate(LEG_QPOS_ADR):
            self.data.qpos[qadr] = self.standing_legs[i]

        for act_id, val in self.upper_act_dict.items():
            self.data.ctrl[act_id] = val

        mujoco.mj_forward(self.model, self.data)

    def key_callback(self, keycode: int):
        speed_step = 0.15
        turn_step = 0.25

        if keycode == KEY_W:
            self.target_vx = min(0.95, self.target_vx + speed_step)
        elif keycode == KEY_S:
            self.target_vx = max(-0.30, self.target_vx - speed_step)
        elif keycode == KEY_A:
            self.target_wz = min(1.0, self.target_wz + turn_step)
        elif keycode == KEY_D:
            self.target_wz = max(-1.0, self.target_wz - turn_step)
        elif keycode == KEY_Q:
            self.target_vy = min(0.30, self.target_vy + 0.10)
        elif keycode == KEY_E:
            self.target_vy = max(-0.30, self.target_vy - 0.10)
        elif keycode == KEY_SPACE:
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_wz = 0.0
        elif keycode == KEY_R:
            self.reset_to_origin()
            print("\n[RESET] Robot repositioned to origin (0, 0)")
        elif keycode == KEY_V:
            self.chase_cam = not self.chase_cam
            print(f"\n[CAMERA] {'Chase Cam (Behind Robot)' if self.chase_cam else 'Free Cam'}")
        elif keycode == KEY_1:
            self.target_vx = 0.40  # Slow walk preset
        elif keycode == KEY_2:
            self.target_vx = 0.75  # Brisk walk preset

    def update_step(self, dt: float):
        # 1. Smooth command filtering (acceleration limits)
        acc_lin = 1.8 * dt
        acc_ang = 3.0 * dt
        self.cmd_vx = np.clip(self.target_vx, self.cmd_vx - acc_lin, self.cmd_vx + acc_lin)
        self.cmd_vy = np.clip(self.target_vy, self.cmd_vy - acc_lin, self.cmd_vy + acc_lin)
        self.cmd_wz = np.clip(self.target_wz, self.cmd_wz - acc_ang, self.cmd_wz + acc_ang)

        speed = math.sqrt(self.cmd_vx**2 + self.cmd_vy**2)
        total_activity = speed + 0.3 * abs(self.cmd_wz)

        # 2. Blend between standing and walking
        if total_activity < 0.04:
            self.stand_blend = min(1.0, self.stand_blend + 4.0 * dt)
        else:
            self.stand_blend = max(0.0, self.stand_blend - 4.0 * dt)

        # 3. Update Heading & World Translation
        self.heading += self.cmd_wz * dt

        # Translation in world frame
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        vx_world = cos_h * self.cmd_vx - sin_h * self.cmd_vy
        vy_world = sin_h * self.cmd_vx + cos_h * self.cmd_vy

        self.pos_x += vx_world * dt
        self.pos_y += vy_world * dt

        # 4. Advance Gait Phase
        if self.stand_blend < 0.99:
            speed_ratio = max(0.2, speed / self.nominal_speed) if speed > 0.05 else 0.5 * abs(self.cmd_wz)
            phase_speed = 60.0 * speed_ratio  # frames per second
            self.gait_phase += phase_speed * dt
            # Cyclic wrapping between cycle_start and cycle_end
            if self.gait_phase >= self.cycle_end:
                self.gait_phase = self.cycle_start + (self.gait_phase - self.cycle_end) % self.cycle_len

        # 5. Interpolate MoCap frame
        f_low = int(math.floor(self.gait_phase))
        f_high = int(math.ceil(self.gait_phase))
        if f_high >= self.cycle_end:
            f_high = self.cycle_start
        alpha = self.gait_phase - f_low

        q_low = self.raw_qpos[f_low]
        q_high = self.raw_qpos[f_high]

        # Interpolated walk joint positions
        walk_legs = (1.0 - alpha) * q_low[LEG_QPOS_ADR] + alpha * q_high[LEG_QPOS_ADR]

        # 6. Apply Active Hip Yaw Steering into turn
        # When turning, steer feet into the turn direction
        steer_bias = float(np.clip(0.35 * self.cmd_wz, -0.45, 0.45))
        # Index 0 is left_hip_yaw, index 6 is right_hip_yaw in canonical order
        walk_legs[0] += steer_bias
        walk_legs[6] += steer_bias

        # 7. Blend Walking & Compliant Standing
        b = self.stand_blend
        final_legs = b * self.standing_legs + (1.0 - b) * walk_legs

        # 8. Set MuJoCo Root Position & Orientation
        self.data.qpos[0] = self.pos_x
        self.data.qpos[1] = self.pos_y
        walk_root_z = (1.0 - alpha) * q_low[2] + alpha * q_high[2]
        self.data.qpos[2] = b * self.standing_root_z + (1.0 - b) * walk_root_z

        # Orientation: combine Heading Yaw with MoCap pelvic roll/pitch
        pelvis_roll = (1.0 - alpha) * q_low[4] + alpha * q_high[4]
        pelvis_pitch = (1.0 - alpha) * q_low[5] + alpha * q_high[5]
        # Quat for heading
        q_heading = sptr.Rotation.from_euler("z", self.heading)
        q_pelvis = sptr.Rotation.from_euler("xy", [pelvis_roll * (1.0 - b), pelvis_pitch * (1.0 - b)])
        full_rot = q_heading * q_pelvis
        quat = full_rot.as_quat() # [x, y, z, w]

        self.data.qpos[3] = quat[3] # w
        self.data.qpos[4] = quat[0] # x
        self.data.qpos[5] = quat[1] # y
        self.data.qpos[6] = quat[2] # z

        # 9. Set Leg Joints
        for i, qadr in enumerate(LEG_QPOS_ADR):
            self.data.qpos[qadr] = final_legs[i]

        # 10. Upper Body Arm Swings synchronized with walking
        arm_swing = (1.0 - b) * 0.35 * (self.cmd_vx / self.nominal_speed) * math.sin(
            2.0 * math.pi * (self.gait_phase - self.cycle_start) / self.cycle_len
        )
        self.data.qpos[9]  = 1.57 - arm_swing  # right_shoulder_pitch
        self.data.qpos[10] = -0.15             # right_shoulder_roll
        self.data.qpos[11] = -0.75             # right_elbow
        self.data.qpos[12] = 1.57 + arm_swing  # left_shoulder_pitch
        self.data.qpos[13] = 0.15              # left_shoulder_roll
        self.data.qpos[14] = -0.75             # left_elbow

        mujoco.mj_forward(self.model, self.data)

    def run_gui(self):
        try:
            import mujoco.viewer
        except ImportError:
            print("[ERROR] mujoco.viewer requires OpenGL/GLFW. Use --render-demo instead.")
            return

        print("=" * 78)
        print("  BARELANG-S SUPERMIMIC STEERING CONTROLLER (WASD Real-Time Teleop)")
        print("=" * 78)
        print("  Controls:")
        print("    [W] / [S] : Forward / Backward / Slow down")
        print("    [A] / [D] : Steer / Turn Left / Right")
        print("    [Q] / [E] : Strafe Left / Right")
        print("    [SPACE]   : Instant Stop (Compliant Standing)")
        print("    [R]       : Reset to Origin (0, 0)")
        print("    [1] / [2] : Speed presets (Slow / Brisk)")
        print("    [V]       : Toggle Chase Cam")
        print("=" * 78)

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")

            last_hud = 0.0
            dt = 1.0 / 60.0

            while viewer.is_running():
                step_start = time.time()

                self.update_step(dt)

                # Camera chase tracking
                if self.chase_cam and torso_bid != -1:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = torso_bid
                    viewer.cam.distance = 2.2
                    viewer.cam.elevation = -14.0
                    # Align camera azimuth behind robot heading
                    viewer.cam.azimuth = math.degrees(self.heading) + 90.0

                viewer.sync()

                # HUD in terminal
                now = time.time()
                if now - last_hud > 0.12:
                    last_hud = now
                    state_str = "STAND" if self.stand_blend > 0.5 else "WALK"
                    hud = (
                        f"\r[{state_str}] Cmd: Vx={self.cmd_vx:+.2f}m/s Vy={self.cmd_vy:+.2f}m/s "
                        f"Wz={self.cmd_wz:+.2f}rad/s | Heading={math.degrees(self.heading):+6.1f}° | "
                        f"Pos: ({self.pos_x:+.2f}m, {self.pos_y:+.2f}m)   "
                    )
                    sys.stdout.write(hud)
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    def render_demo(self, output_path: str = "data/mocap/barelang_steering_demo.mp4", duration_sec: float = 8.0):
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) required for video rendering.")
            return

        print(f"\nRendering steering demonstration ({duration_sec}s @ 60fps) to: {output_path}")
        renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        if torso_bid != -1:
            camera.trackbodyid = torso_bid
        camera.distance = 2.2
        camera.elevation = -15.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, 60.0, (640, 480))

        total_frames = int(duration_sec * 60)
        dt = 1.0 / 60.0

        self.reset_to_origin()

        for f in range(total_frames):
            t = f * dt
            # Pre-programmed steering trajectory:
            # 0.0s - 1.0s: Stand still then start walking
            # 1.0s - 3.0s: Walk straight (Vx = 0.7)
            # 3.0s - 5.0s: Turn left while walking (Vx = 0.65, Wz = +0.55 rad/s)
            # 5.0s - 7.0s: Turn right while walking (Vx = 0.65, Wz = -0.55 rad/s)
            # 7.0s - 8.0s: Stop smoothly to compliant stand (Vx = 0, Wz = 0)
            if t < 0.8:
                self.target_vx = 0.0
                self.target_wz = 0.0
            elif t < 3.0:
                self.target_vx = 0.70
                self.target_wz = 0.0
            elif t < 5.0:
                self.target_vx = 0.65
                self.target_wz = 0.55
            elif t < 7.0:
                self.target_vx = 0.65
                self.target_wz = -0.55
            else:
                self.target_vx = 0.0
                self.target_wz = 0.0

            self.update_step(dt)

            camera.azimuth = math.degrees(self.heading) + 90.0
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if f % 30 == 0 or f == total_frames - 1:
                print(f"\rRendered {f+1}/{total_frames} frames ({f/total_frames*100:.1f}%) | "
                      f"Pos: ({self.pos_x:.2f}m, {self.pos_y:.2f}m) Heading: {math.degrees(self.heading):.1f}°", end="", flush=True)

        writer.release()
        print(f"\n[OK] Steering demo video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Steering for BarelangS with SuperMimic")
    parser.add_argument("--mocap", type=str, default="data/mocap/barelang_mocap_walk.npz")
    parser.add_argument("--xml", type=str, default="barelangS_terrain_mujoco.xml")
    parser.add_argument("--render-demo", type=str, default=None, help="Render an MP4 video of steering")
    parser.add_argument("--duration", type=float, default=8.0, help="Demo duration in seconds")
    args = parser.parse_args()

    controller = SuperMimicSteeringController(mocap_path=args.mocap, xml_path=args.xml)

    if args.render_demo:
        controller.render_demo(output_path=args.render_demo, duration_sec=args.duration)
    else:
        controller.run_gui()

