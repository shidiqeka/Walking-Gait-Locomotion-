#!/usr/bin/env python3
"""
steer_barelang_supermimic.py
----------------------------
Ultimate Multi-Skill Locomotion Controller for BarelangS Humanoid
Powered by SuperMimic and Bio-mechanical Human Motion Capture.

Capabilities:
  - Compliant Standing (Idle / Standby)
  - Human Walk (0.3 - 0.8 m/s)
  - Human High-Speed Sprint / Run (0.8 - 1.62 m/s with flight phase)
  - Dynamic Soccer Kick (Right Foot Strike)
  - Omnidirectional Turning & Strafing (WASD)
  - Chase-Cam & Free-Cam Tracking

Keyboard Controls (In MuJoCo Viewer):
  [W]          : Increase Forward Speed (Walk -> Sprint)
  [S]          : Decelerate / Walk Backward / Stop
  [A] / [D]    : Steer / Turn Left / Right
  [Q] / [E]    : Strafe Left / Right
  [K]          : Dynamic Soccer Kick (Right Foot)
  [SPACE]      : Instant Stop / Stand Still
  [1]          : Preset: Walk (0.50 m/s)
  [2]          : Preset: Fast Walk (0.80 m/s)
  [3]          : Preset: Sprint / Run (1.55 m/s)
  [V]          : Toggle Camera Mode (Chase Cam vs Orbit Cam)
  [R]          : Reset to Origin (0, 0)
  [ESC]        : Exit

Usage:
  # Interactive 3D Window with WASD controls:
  python steer_barelang_supermimic.py

  # Headless video demonstration (Walk -> Sprint -> Turn -> Kick -> Stand):
  python steer_barelang_supermimic.py --render-demo data/mocap/barelang_multiskill_demo.mp4
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
KEY_K = 75
KEY_R = 82
KEY_V = 86
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51

LEG_QPOS_ADR = [21, 22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20]
LEG_ACT_ID   = [12, 0, 1, 2, 3, 4, 13, 5, 6, 7, 8, 9]

class MultiSkillController:
    def __init__(
        self,
        walk_path: str = "data/mocap/barelang_mocap_walk.npz",
        run_path: str = "data/mocap/barelang_mocap_run.npz",
        xml_path: str = "barelangS_terrain_mujoco.xml",
    ):
        print("Loading Walk & Run MoCap libraries...")
        w_data = np.load(walk_path)
        r_data = np.load(run_path)

        self.walk_qpos = w_data["qpos"].astype(np.float32)
        self.walk_cycle_start = int(w_data.get("cycle_start", 18))
        self.walk_cycle_end = int(w_data.get("cycle_end", 80))
        self.walk_cycle_len = self.walk_cycle_end - self.walk_cycle_start
        self.walk_speed_ref = float(w_data.get("walking_speed", 0.710))

        self.run_qpos = r_data["qpos"].astype(np.float32)
        self.run_cycle_start = int(r_data.get("cycle_start", 14))
        self.run_cycle_end = int(r_data.get("cycle_end", 60))
        self.run_cycle_len = self.run_cycle_end - self.run_cycle_start
        self.run_speed_ref = float(r_data.get("walking_speed", 1.621))

        print(f"Loading MuJoCo physics model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Compliant Standing Pose
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

        # Locomotion Command & State
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0

        self.pos_x = 0.0
        self.pos_y = 0.0
        self.heading = 0.0
        self.walk_phase = float(self.walk_cycle_start)
        self.run_phase = float(self.run_cycle_start)

        self.stand_blend = 1.0  # 1.0 = stand, 0.0 = motion
        self.run_blend = 0.0    # 0.0 = walk, 1.0 = sprint/run

        # Kick state
        self.is_kicking = False
        self.kick_progress = 0.0  # 0.0 to 1.0
        self.kick_duration = 0.65 # seconds

        self.chase_cam = True
        self.reset_requested = False
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
        self.walk_phase = float(self.walk_cycle_start)
        self.run_phase = float(self.run_cycle_start)
        self.stand_blend = 1.0
        self.run_blend = 0.0
        self.is_kicking = False
        self.kick_progress = 0.0
        self.reset_requested = False

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.standing_root_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.qvel[:] = 0.0

        for i, qadr in enumerate(LEG_QPOS_ADR):
            self.data.qpos[qadr] = self.standing_legs[i]

        for act_id, val in self.upper_act_dict.items():
            self.data.ctrl[act_id] = val

        mujoco.mj_forward(self.model, self.data)

    def trigger_kick(self):
        if not self.is_kicking:
            self.is_kicking = True
            self.kick_progress = 0.0
            print("\n[ACTION] DYNAMIC SOCCER KICK!")

    def key_callback(self, keycode: int):
        speed_step = 0.18
        turn_step = 0.25

        if keycode == KEY_W:
            self.target_vx = min(1.65, self.target_vx + speed_step)
        elif keycode == KEY_S:
            self.target_vx = max(-0.35, self.target_vx - speed_step)
        elif keycode == KEY_A:
            self.target_wz = min(1.0, self.target_wz + turn_step)
        elif keycode == KEY_D:
            self.target_wz = max(-1.0, self.target_wz - turn_step)
        elif keycode == KEY_Q:
            self.target_vy = min(0.35, self.target_vy + 0.10)
        elif keycode == KEY_E:
            self.target_vy = max(-0.35, self.target_vy - 0.10)
        elif keycode == KEY_K:
            self.trigger_kick()
        elif keycode == KEY_SPACE:
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_wz = 0.0
            self.is_kicking = False
        elif keycode == KEY_1:
            self.target_vx = 0.50  # Walk
        elif keycode == KEY_2:
            self.target_vx = 0.80  # Fast Walk
        elif keycode == KEY_3:
            self.target_vx = 1.55  # Sprint / Run
        elif keycode in [KEY_R, 82, 114]:
            self.reset_requested = True
            print("\n[RESET REQUESTED]")
        elif keycode == KEY_V:
            self.chase_cam = not self.chase_cam
            print(f"\n[CAMERA] {'Chase Cam' if self.chase_cam else 'Free Cam'}")

    def compute_kick_pose(self) -> np.ndarray:
        """Computes 12-leg joint positions during soccer kick."""
        t = self.kick_progress
        legs = self.standing_legs.copy()

        if t < 0.30:  # Windup: shift weight to left stance, draw right leg back
            s = t / 0.30
            legs[1] -= 0.14 * s  # L hip roll (weight shift)
            legs[8] += 0.50 * s  # R hip pitch back
            legs[9] += 0.95 * s  # R knee bend
        elif t < 0.60:  # Explosive forward strike
            s = (t - 0.30) / 0.30
            legs[1] -= 0.14
            legs[8] = 0.50 - 1.35 * s   # R hip pitch forward to -0.85
            legs[9] = 1.25 - 1.05 * s   # R knee snaps forward to 0.20
            legs[10] = -0.10 - 0.25 * s # R ankle stiff
        else:  # Follow-through and recovery
            s = (t - 0.60) / 0.40
            legs[1] = -0.14 * (1.0 - s)
            legs[8] = -0.85 * (1.0 - s) + (-0.20) * s
            legs[9] = 0.20 * (1.0 - s) + 0.30 * s
            legs[10] = -0.35 * (1.0 - s) + (-0.10) * s

        return legs

    def update_step(self, dt: float):
        if self.reset_requested:
            self.reset_to_origin()
            print("[RESET] Returned to origin (0, 0)")
            return

        # 1. Smooth command filtering
        acc_lin = 2.2 * dt
        acc_ang = 3.5 * dt
        self.cmd_vx = np.clip(self.target_vx, self.cmd_vx - acc_lin, self.cmd_vx + acc_lin)
        self.cmd_vy = np.clip(self.target_vy, self.cmd_vy - acc_lin, self.cmd_vy + acc_lin)
        self.cmd_wz = np.clip(self.target_wz, self.cmd_wz - acc_ang, self.cmd_wz + acc_ang)

        speed = math.sqrt(self.cmd_vx**2 + self.cmd_vy**2)

        # 2. Stand vs Walk vs Run Blending
        if speed < 0.05 and abs(self.cmd_wz) < 0.05 and not self.is_kicking:
            self.stand_blend = min(1.0, self.stand_blend + 4.5 * dt)
        else:
            self.stand_blend = max(0.0, self.stand_blend - 4.5 * dt)

        # Run blend: 0.0 when speed <= 0.75, 1.0 when speed >= 1.35
        target_run_b = np.clip((speed - 0.75) / (1.35 - 0.75), 0.0, 1.0)
        self.run_blend += np.clip(target_run_b - self.run_blend, -3.0 * dt, 3.0 * dt)

        # 3. Update Heading & World Translation
        self.heading += self.cmd_wz * dt
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        vx_world = cos_h * self.cmd_vx - sin_h * self.cmd_vy
        vy_world = sin_h * self.cmd_vx + cos_h * self.cmd_vy
        self.pos_x += vx_world * dt
        self.pos_y += vy_world * dt

        # 4. Advance Gait Phases
        if self.stand_blend < 0.99 and not self.is_kicking:
            # Walk phase
            w_speed_ratio = max(0.2, min(1.1, speed / self.walk_speed_ref)) if speed > 0.05 else 0.5 * abs(self.cmd_wz)
            self.walk_phase += 60.0 * w_speed_ratio * dt
            if self.walk_phase >= self.walk_cycle_end:
                self.walk_phase = self.walk_cycle_start + (self.walk_phase - self.walk_cycle_end) % self.walk_cycle_len

            # Run phase
            r_speed_ratio = max(0.6, speed / self.run_speed_ref)
            self.run_phase += 60.0 * r_speed_ratio * dt
            if self.run_phase >= self.run_cycle_end:
                self.run_phase = self.run_cycle_start + (self.run_phase - self.run_cycle_end) % self.run_cycle_len

        # 5. Interpolate MoCap Poses
        # Walk pose
        f_w0 = int(math.floor(self.walk_phase))
        f_w1 = (f_w0 + 1) if (f_w0 + 1) < self.walk_cycle_end else self.walk_cycle_start
        a_w = self.walk_phase - f_w0
        walk_legs = (1.0 - a_w) * self.walk_qpos[f_w0, LEG_QPOS_ADR] + a_w * self.walk_qpos[f_w1, LEG_QPOS_ADR]
        walk_root_z = (1.0 - a_w) * self.walk_qpos[f_w0, 2] + a_w * self.walk_qpos[f_w1, 2]

        # Run pose
        f_r0 = int(math.floor(self.run_phase))
        f_r1 = (f_r0 + 1) if (f_r0 + 1) < self.run_cycle_end else self.run_cycle_start
        a_r = self.run_phase - f_r0
        run_legs = (1.0 - a_r) * self.run_qpos[f_r0, LEG_QPOS_ADR] + a_r * self.run_qpos[f_r1, LEG_QPOS_ADR]
        run_root_z = (1.0 - a_r) * self.run_qpos[f_r0, 2] + a_r * self.run_qpos[f_r1, 2]

        # Blend Walk & Run
        r_b = self.run_blend
        loco_legs = (1.0 - r_b) * walk_legs + r_b * run_legs
        loco_root_z = (1.0 - r_b) * walk_root_z + r_b * run_root_z

        # Steering bias into turn
        steer_bias = float(np.clip(0.35 * self.cmd_wz, -0.45, 0.45))
        loco_legs[0] += steer_bias  # L hip yaw
        loco_legs[6] += steer_bias  # R hip yaw

        # 6. Kick Action Processing
        if self.is_kicking:
            kick_legs = self.compute_kick_pose()
            final_legs = kick_legs
            final_root_z = self.standing_root_z
            self.kick_progress += dt / self.kick_duration
            if self.kick_progress >= 1.0:
                self.is_kicking = False
                self.kick_progress = 0.0
                print("\n[ACTION] Kick completed! Returned to stance.")
        else:
            # Blend Locomotion & Standing
            s_b = self.stand_blend
            final_legs = s_b * self.standing_legs + (1.0 - s_b) * loco_legs
            final_root_z = s_b * self.standing_root_z + (1.0 - s_b) * loco_root_z

        # 7. Apply to MuJoCo data
        self.data.qpos[0] = self.pos_x
        self.data.qpos[1] = self.pos_y
        self.data.qpos[2] = final_root_z

        # Orientation: heading yaw + pelvic tilt
        q_heading = sptr.Rotation.from_euler("z", self.heading)
        self.data.qpos[3:7] = [q_heading.as_quat()[3], q_heading.as_quat()[0], q_heading.as_quat()[1], q_heading.as_quat()[2]]

        for i, qadr in enumerate(LEG_QPOS_ADR):
            self.data.qpos[qadr] = final_legs[i]

        # Arms: dynamic arm swing scales with speed and run blend
        arm_amp = 0.35 * (1.0 - r_b) + 0.65 * r_b
        arm_pitch_offset = -0.40 * r_b  # lean forward during sprint
        arm_swing = (1.0 - self.stand_blend) * arm_amp * math.sin(
            2.0 * math.pi * (self.run_phase if r_b > 0.5 else self.walk_phase) / 60.0
        )
        elbow = -0.75 * (1.0 - r_b) - 1.25 * r_b

        self.data.qpos[9]  = 1.57 - arm_swing + arm_pitch_offset
        self.data.qpos[10] = -0.15
        self.data.qpos[11] = elbow
        self.data.qpos[12] = 1.57 + arm_swing + arm_pitch_offset
        self.data.qpos[13] = 0.15
        self.data.qpos[14] = elbow

        mujoco.mj_forward(self.model, self.data)

    def run_gui(self):
        try:
            import mujoco.viewer
        except ImportError:
            print("[ERROR] mujoco.viewer requires OpenGL/GLFW. Use --render-demo instead.")
            return

        print("=" * 78)
        print("  BARELANG-S SUPERMIMIC MULTI-SKILL CONTROLLER")
        print("  [Stand | Walk | Sprint / Run | Dynamic Soccer Kick]")
        print("=" * 78)
        print("  Controls:")
        print("    [W] / [S] : Forward / Backward (Walk -> Sprint)")
        print("    [A] / [D] : Steer Left / Right")
        print("    [Q] / [E] : Strafe Left / Right")
        print("    [K]       : DYNAMIC SOCCER KICK (Right Foot)")
        print("    [SPACE]   : Instant Stop / Compliant Stand")
        print("    [1]       : Walk Preset (0.50 m/s)")
        print("    [2]       : Fast Walk Preset (0.80 m/s)")
        print("    [3]       : SPRINT / RUN Preset (1.55 m/s)")
        print("    [V]       : Toggle Chase Cam")
        print("    [R]       : Reset to Origin")
        print("=" * 78)

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")

            last_hud = 0.0
            dt = 1.0 / 60.0

            while viewer.is_running():
                step_start = time.time()
                self.update_step(dt)

                if self.chase_cam and torso_bid != -1:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = torso_bid
                    viewer.cam.distance = 2.4
                    viewer.cam.elevation = -14.0
                    viewer.cam.azimuth = math.degrees(self.heading) + 90.0

                viewer.sync()

                now = time.time()
                if now - last_hud > 0.12:
                    last_hud = now
                    if self.is_kicking:
                        gait_str = f"SOCCER KICK ({self.kick_progress*100:3.0f}%)"
                    elif self.stand_blend > 0.5:
                        gait_str = "STAND"
                    elif self.run_blend > 0.5:
                        gait_str = f"SPRINT ({self.cmd_vx:.2f}m/s)"
                    else:
                        gait_str = f"WALK ({self.cmd_vx:.2f}m/s)"

                    hud = (
                        f"\r[{gait_str:18s}] Heading={math.degrees(self.heading):+5.1f}° | "
                        f"Pos: ({self.pos_x:+.2f}m, {self.pos_y:+.2f}m) | Speed: {self.cmd_vx:.2f}m/s  "
                    )
                    sys.stdout.write(hud)
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    def render_demo(self, output_path: str = "data/mocap/barelang_multiskill_demo.mp4", duration_sec: float = 10.0):
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) required for video rendering.")
            return

        print(f"\nRendering Multi-Skill demonstration ({duration_sec}s @ 60fps) to: {output_path}")
        renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        if torso_bid != -1:
            camera.trackbodyid = torso_bid
        camera.distance = 2.5
        camera.elevation = -14.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, 60.0, (640, 480))

        total_frames = int(duration_sec * 60)
        dt = 1.0 / 60.0
        self.reset_to_origin()

        # Multi-Skill Demonstration Timeline:
        # 0.0s - 1.0s: Stand still
        # 1.0s - 3.5s: Smooth transition into Human Walk (Vx = 0.65 m/s)
        # 3.5s - 6.0s: Accelerate into HIGH-SPEED SPRINT / RUN (Vx = 1.55 m/s, 5.6 km/h)
        # 6.0s - 7.5s: Decelerate & execute S-Curve steering turn
        # 7.5s - 8.5s: Stop & execute DYNAMIC SOCCER KICK!
        # 8.5s - 10.0s: Return to rock-solid compliant standing

        for f in range(total_frames):
            t = f * dt
            if t < 1.0:
                self.target_vx = 0.0
                self.target_wz = 0.0
            elif t < 3.5:
                self.target_vx = 0.65
                self.target_wz = 0.0
            elif t < 6.0:
                self.target_vx = 1.55  # Sprint / Run!
                self.target_wz = 0.0
            elif t < 7.3:
                self.target_vx = 0.40
                self.target_wz = 0.60  # Turn into kick position
            elif t < 7.5:
                self.target_vx = 0.0
                self.target_wz = 0.0
            elif t < 8.3:
                if not self.is_kicking and self.kick_progress == 0.0:
                    self.trigger_kick()
            else:
                self.target_vx = 0.0
                self.target_wz = 0.0

            self.update_step(dt)

            camera.azimuth = math.degrees(self.heading) + 90.0
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()

            # Burn HUD into video frame
            if self.is_kicking:
                state_lbl = f"SOCCER KICK ({self.kick_progress*100:.0f}%)"
            elif self.stand_blend > 0.5:
                state_lbl = "COMPLIANT STAND"
            elif self.run_blend > 0.5:
                state_lbl = f"SPRINT / RUN ({self.cmd_vx:.2f} m/s | 5.6 km/h)"
            else:
                state_lbl = f"NATURAL WALK ({self.cmd_vx:.2f} m/s)"

            hud_text = f"T={t:4.1f}s | {state_lbl} | Pos: ({self.pos_x:+.2f}m, {self.pos_y:+.2f}m)"
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

            writer.write(frame)

            if f % 30 == 0 or f == total_frames - 1:
                print(f"\rRendered {f+1}/{total_frames} frames ({f/total_frames*100:.1f}%) | "
                      f"Mode: {state_lbl:30s}", end="", flush=True)

        writer.release()
        print(f"\n[OK] Multi-skill demo video saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BarelangS Multi-Skill Controller (Walk, Sprint, Kick, Steer)")
    parser.add_argument("--walk", type=str, default="data/mocap/barelang_mocap_walk.npz")
    parser.add_argument("--run", type=str, default="data/mocap/barelang_mocap_run.npz")
    parser.add_argument("--xml", type=str, default="barelangS_terrain_mujoco.xml")
    parser.add_argument("--render-demo", type=str, default=None, help="Render an MP4 video of multi-skill demonstration")
    parser.add_argument("--duration", type=float, default=10.0, help="Demo duration in seconds")
    args = parser.parse_args()

    controller = MultiSkillController(walk_path=args.walk, run_path=args.run, xml_path=args.xml)

    if args.render_demo:
        controller.render_demo(output_path=args.render_demo, duration_sec=args.duration)
    else:
        controller.run_gui()
