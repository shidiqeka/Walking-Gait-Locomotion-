#!/usr/bin/env python3
"""
test_push_recovery.py
---------------------
Interactive Push Recovery & Anti-Fall Simulator for BarelangS Humanoid
Powered by SuperMimic and Bio-mechanical Capture Point Stepping Strategy.

Keyboard Controls (In MuJoCo Viewer):
  [I]          : Push Robot from Back (Forward Shove: +25 N)
  [K]          : Push Robot from Front (Backward Shove: -25 N)
  [J]          : Push Robot from Right (Lateral Shove: +20 N)
  [L]          : Push Robot from Left (Lateral Shove: -20 N)
  [P]          : Random Heavy Push (30 N in random direction)
  [W] / [S]    : Walk Forward / Slow Down
  [A] / [D]    : Steer Left / Right
  [SPACE]      : Toggle Walk / Stand
  [R]          : Reset to Origin (0, 0)
  [ESC]        : Exit

Usage:
  # Interactive 3D Window (GLFW):
  python test_push_recovery.py

  # Headless video demonstration of push recovery:
  python test_push_recovery.py --render-demo data/mocap/barelang_push_recovery_demo.mp4
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import scipy.spatial.transform as sptr
import mujoco

from supermimic.push_recovery_controller import PushRecoveryController

# GLFW Key Constants
KEY_SPACE = 32
KEY_W = 87
KEY_S = 83
KEY_A = 65
KEY_D = 68
KEY_R = 82
KEY_I = 73
KEY_K = 75
KEY_J = 74
KEY_L = 76
KEY_P = 80

# Canonical Leg Address and Actuator IDs
LEG_QPOS_ADR = [21, 22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20]
LEG_ACT_ID   = [12, 0, 1, 2, 3, 4, 13, 5, 6, 7, 8, 9]

class PushRecoverySimulator:
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
        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")

        self.recovery = PushRecoveryController(leg_length=0.38)

        # Compliant Standing Pose (qpos for 12 legs in canonical order)
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

        # Substepping parameters
        self.sim_dt = self.model.opt.timestep
        self.control_dt = 1.0 / 60.0
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))

        # Push disturbance variables
        self.push_force = np.zeros(3, dtype=np.float32)
        self.push_steps_remaining = 0
        self.active_push_label = "None"

        # Locomotion state
        self.cmd_vx = 0.50
        self.cmd_wz = 0.0
        self.gait_phase = float(self.cycle_start)
        self.stand_blend = 0.0
        self.is_walking = True

        self.reset_to_start()

    def reset_to_start(self):
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

        self.gait_phase = float(self.cycle_start)
        self.stand_blend = 0.0
        self.push_steps_remaining = 0
        self.push_force[:] = 0.0
        self.active_push_label = "None"

        mujoco.mj_forward(self.model, self.data)

    def apply_push(self, fx: float, fy: float, duration_sec: float = 0.12, label: str = "PUSH"):
        """Applies external force impulse to robot torso."""
        self.push_force[0] = fx
        self.push_force[1] = fy
        self.push_force[2] = 0.0
        # Convert duration to control steps
        self.push_steps_remaining = int(round(duration_sec / self.control_dt))
        self.active_push_label = f"{label} ({fx:+.0f}N X, {fy:+.0f}N Y)"
        print(f"\n[DISTURBANCE] Applied {self.active_push_label}!")

    def key_callback(self, keycode: int):
        if keycode == KEY_I:
            self.apply_push(fx=+26.0, fy=0.0, label="Back Push (Shove Forward)")
        elif keycode == KEY_K:
            self.apply_push(fx=-26.0, fy=0.0, label="Front Push (Shove Backward)")
        elif keycode == KEY_J:
            self.apply_push(fx=0.0, fy=+22.0, label="Right Push (Shove Left)")
        elif keycode == KEY_L:
            self.apply_push(fx=0.0, fy=-22.0, label="Left Push (Shove Right)")
        elif keycode == KEY_P:
            # Random impulse in 360 degrees
            angle = np.random.uniform(0, 2 * math.pi)
            mag = 30.0
            fx = mag * math.cos(angle)
            fy = mag * math.sin(angle)
            self.apply_push(fx=fx, fy=fy, label=f"Random Shock ({mag}N @ {math.degrees(angle):.0f}°)")
        elif keycode == KEY_SPACE:
            self.is_walking = not self.is_walking
            print(f"\n[STATE] {'WALKING' if self.is_walking else 'STANDING'}")
        elif keycode == KEY_W:
            self.cmd_vx = min(0.9, self.cmd_vx + 0.15)
        elif keycode == KEY_S:
            self.cmd_vx = max(0.0, self.cmd_vx - 0.15)
        elif keycode == KEY_A:
            self.cmd_wz = min(0.8, self.cmd_wz + 0.20)
        elif keycode == KEY_D:
            self.cmd_wz = max(-0.8, self.cmd_wz - 0.20)
        elif keycode == KEY_R:
            self.reset_to_start()
            print("\n[RESET] Simulation reset to origin")

    def step(self):
        dt = self.control_dt

        # 1. Read IMU states from MuJoCo simulation
        quat = self.data.qpos[3:7].copy() # [w, x, y, z]
        w, x, y, z = quat
        R = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
            [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)]
        ], dtype=np.float32)

        # Gyro in base frame
        ang_vel_world = self.data.qvel[3:6]
        lin_vel_world = self.data.qvel[0:3]
        gyro_base = R.T @ ang_vel_world
        lin_vel_base = R.T @ lin_vel_world

        self.recovery.update_imu(quat, gyro_base, lin_vel_base, dt)

        # 2. Determine swing leg from gait cycle
        # Cycle frames 18 to 49 ~ Right leg swing, frames 49 to 80 ~ Left leg swing
        rel_phase = (self.gait_phase - self.cycle_start) % self.cycle_len
        swing_leg = "right" if rel_phase < (self.cycle_len / 2.0) else "left"

        # 3. Compute Bio-mechanical Recovery Offsets
        rec_data = self.recovery.compute_recovery_offsets(swing_leg)
        leg_offsets = rec_data["leg_offsets"]
        arm_p_off = rec_data["arm_pitch_offset"]
        arm_r_off = rec_data["arm_roll_offset"]
        phase_accel = rec_data["phase_acceleration"]

        # 4. Advance Gait Phase with Early Touchdown modulation
        if self.is_walking:
            self.stand_blend = max(0.0, self.stand_blend - 4.0 * dt)
            phase_speed = 60.0 * (self.cmd_vx / self.nominal_speed) * phase_accel
            self.gait_phase += phase_speed * dt
            if self.gait_phase >= self.cycle_end:
                self.gait_phase = self.cycle_start + (self.gait_phase - self.cycle_end) % self.cycle_len
        else:
            self.stand_blend = min(1.0, self.stand_blend + 4.0 * dt)

        # 5. Interpolate MoCap Reference
        f_low = int(math.floor(self.gait_phase))
        f_high = int(math.ceil(self.gait_phase))
        if f_high >= self.cycle_end:
            f_high = self.cycle_start
        alpha = self.gait_phase - f_low

        ref_legs = (1.0 - alpha) * self.raw_qpos[f_low, LEG_QPOS_ADR] + alpha * self.raw_qpos[f_high, LEG_QPOS_ADR]

        # Blend walking and standing
        b = self.stand_blend
        base_targets = (1.0 - b) * ref_legs + b * self.standing_legs

        # 6. Apply Active Recovery Joint Offsets
        final_targets = base_targets + leg_offsets

        # 7. Apply controls to actuators
        for i, act_id in enumerate(LEG_ACT_ID):
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(final_targets[i], ctrl_min, ctrl_max)

        # Arm dynamic damping
        arm_swing = (1.0 - b) * 0.35 * math.sin(2.0 * math.pi * rel_phase / self.cycle_len)
        for act_id, val in self.upper_act_dict.items():
            if act_id == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch"):
                self.data.ctrl[act_id] = val - arm_swing + arm_p_off
            elif act_id == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch"):
                self.data.ctrl[act_id] = val + arm_swing + arm_p_off
            elif act_id == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_roll"):
                self.data.ctrl[act_id] = val - arm_r_off
            elif act_id == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_roll"):
                self.data.ctrl[act_id] = val + arm_r_off
            else:
                self.data.ctrl[act_id] = val

        # 8. Apply Push Disturbances if active
        if self.push_steps_remaining > 0:
            self.data.xfrc_applied[self.torso_bid, :3] = self.push_force
            self.push_steps_remaining -= 1
            if self.push_steps_remaining == 0:
                self.data.xfrc_applied[self.torso_bid, :] = 0.0
                self.active_push_label = "None"

        # 9. Step MuJoCo physical integration
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def run_gui(self):
        try:
            import mujoco.viewer
        except ImportError:
            print("[ERROR] mujoco.viewer requires OpenGL/GLFW. Use --render-demo instead.")
            return

        print("=" * 78)
        print("  BARELANG-S SUPERMIMIC PUSH RECOVERY & ANTI-FALL TESTER")
        print("=" * 78)
        print("  Push Controls:")
        print("    [I] : Push from Back (Forward Shove: +26 N)")
        print("    [K] : Push from Front (Backward Shove: -26 N)")
        print("    [J] : Push from Right (Lateral Shove: +22 N)")
        print("    [L] : Push from Left (Lateral Shove: -22 N)")
        print("    [P] : Random Heavy Shock (30 N)")
        print("    [SPACE] : Walk / Stand Toggle")
        print("    [R] : Reset")
        print("=" * 78)

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            if self.torso_bid != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = self.torso_bid
                viewer.cam.distance = 2.2
                viewer.cam.elevation = -14.0
                viewer.cam.azimuth = 125.0

            last_hud = 0.0

            while viewer.is_running():
                step_start = time.time()

                self.step()
                viewer.sync()

                now = time.time()
                if now - last_hud > 0.12:
                    last_hud = now
                    p_deg = math.degrees(self.recovery.pitch)
                    r_deg = math.degrees(self.recovery.roll)
                    status = "RECOVERING" if self.recovery.is_disturbed else "STABLE"
                    hud = (
                        f"\r[{status:10s}] Pitch: {p_deg:+5.1f}° Roll: {r_deg:+5.1f}° | "
                        f"Robot X: {self.data.qpos[0]:.2f}m | Push: {self.active_push_label:28s} "
                    )
                    sys.stdout.write(hud)
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)

    def render_demo(self, output_path: str = "data/mocap/barelang_push_recovery_demo.mp4", duration_sec: float = 8.5):
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) required for video rendering.")
            return

        print(f"\nRendering Push Recovery demonstration ({duration_sec}s @ 60fps) to: {output_path}")
        renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        if self.torso_bid != -1:
            camera.trackbodyid = self.torso_bid
        camera.distance = 2.2
        camera.elevation = -14.0
        camera.azimuth = 125.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, 60.0, (640, 480))

        total_frames = int(duration_sec * 60)
        self.reset_to_start()

        # Scheduled Perturbations:
        # At 1.5s: Heavy Back Push (+26 N) -> Robot steps forward to catch itself
        # At 3.8s: Front Push (-24 N) -> Robot takes recovery step backward
        # At 6.0s: Lateral Push (+22 N) -> Robot widens stance & arm counter-damps
        push_schedule = [
            (1.5, +26.0, 0.0, "BACK PUSH (+26N)"),
            (3.8, -24.0, 0.0, "FRONT PUSH (-24N)"),
            (6.0, 0.0, +22.0, "LATERAL PUSH (+22N)")
        ]

        for f in range(total_frames):
            t = f * self.control_dt

            # Check push triggers
            for t_push, fx, fy, lbl in push_schedule:
                if abs(t - t_push) < (self.control_dt / 2.0):
                    self.apply_push(fx, fy, duration_sec=0.14, label=lbl)

            self.step()

            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()

            # Burn HUD overlay text into video frame using OpenCV
            p_deg = math.degrees(self.recovery.pitch)
            r_deg = math.degrees(self.recovery.roll)
            hud_text = f"T={t:.2f}s | Pitch={p_deg:+.1f} Roll={r_deg:+.1f} | Push: {self.active_push_label}"
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            writer.write(frame)

            if f % 30 == 0 or f == total_frames - 1:
                print(f"\rRendered {f+1}/{total_frames} frames ({f/total_frames*100:.1f}%) | "
                      f"Robot X: {self.data.qpos[0]:.2f}m | Status: {'RECOVER' if self.recovery.is_disturbed else 'STABLE'}", end="", flush=True)

        writer.release()
        print(f"\n[OK] Push recovery demo video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BarelangS Push Recovery & Anti-Fall Simulator")
    parser.add_argument("--mocap", type=str, default="data/mocap/barelang_mocap_walk.npz")
    parser.add_argument("--xml", type=str, default="barelangS_terrain_mujoco.xml")
    parser.add_argument("--render-demo", type=str, default=None, help="Render an MP4 video of push recovery")
    parser.add_argument("--duration", type=float, default=8.5, help="Demo duration in seconds")
    args = parser.parse_args()

    sim = PushRecoverySimulator(mocap_path=args.mocap, xml_path=args.xml)

    if args.render_demo:
        sim.render_demo(output_path=args.render_demo, duration_sec=args.duration)
    else:
        sim.run_gui()

