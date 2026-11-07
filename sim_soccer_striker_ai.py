#!/usr/bin/env python3
"""
sim_soccer_striker_ai.py
-------------------------
Autonomous RoboCup Striker AI Simulation for BarelangS Humanoid Robot.
Closed-Loop Newton Physics, Dynamic Contact Mechanics, and Bio-Mechanical Retargeting.

Capabilities:
  - RoboCup Green Pitch Arena with White Regulation Markings.
  - 3D Regulation Goal Posts, Crossbar, and Net at (4.0m, 0.0m).
  - Physical High-Visibility Soccer Ball (Freejoint Newton dynamics).
  - Autonomous Striker Finite State Machine (FSM):
      1. APPROACH  : Face ball and accelerate into High-Speed Sprint (1.4 m/s) -> Walk.
      2. ALIGN     : Orbit behind ball aligned directly with line-of-shot to goal.
      3. STRIKE    : Explosive Bio-Mechanical 0.65s Human Kick launching ball at 10-14 km/h.
      4. EVALUATE  : Goal line detection (X >= 4.0m, |Y| <= 0.9m).
      5. CELEBRATE : Victory celebration with arms raised in 'V' pose upon scoring!
  - Manual Teleoperation Override (WASD, Q/E, K, Space).
  - Thread-Safe Instant Reset ([R]) and Ball Respawn ([B]).
  - Multi-Camera Views ([V]): Dynamic Chase Cam, TV Broadcast Cam, Goal Cam.

Controls (In MuJoCo Viewer):
  [T]          : Toggle Autonomous Striker AI (RoboCup AI Mode)
  [W] / [S]    : Forward / Backward (Walk -> Sprint)
  [A] / [D]    : Turn Left / Right
  [Q] / [E]    : Strafe Left / Right
  [K] / [SPACE]: Dynamic Soccer Kick
  [SPACE]      : Stop / Compliant Stand
  [B]          : Respawn Soccer Ball
  [R]          : Reset Robot and Ball to Origin
  [V]          : Cycle Camera Views (Chase Cam -> Broadcast -> Goal Cam)
  [ESC]        : Exit

Usage:
  # Interactive 3D Physics Window:
  python sim_soccer_striker_ai.py

  # Headless Video Demonstration (Autonomous Match Clip):
  python sim_soccer_striker_ai.py --render-demo data/mocap/barelang_soccer_striker_match.mp4
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
KEY_T = 84
KEY_B = 66
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51

LEG_QPOS_ADR = [21, 22, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20]
LEG_ACT_ID   = [12, 0, 1, 2, 3, 4, 13, 5, 6, 7, 8, 9]

def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

class SoccerStrikerSimulation:
    def __init__(
        self,
        walk_path: str = "data/mocap/barelang_mocap_walk.npz",
        run_path: str = "data/mocap/barelang_mocap_run.npz",
        xml_base_path: str = "barelangS_terrain_mujoco.xml",
        ball_pos: tuple = (2.0, 0.20),
    ):
        print("Loading Human Walking & Sprinting MoCap libraries...")
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

        # Build Soccer Pitch Arena with Regulation Goal Posts & Ball
        print(f"Building RoboCup Soccer Arena from: {xml_base_path}")
        with open(xml_base_path, "r") as f:
            xml_content = f.read()

        xml_flat = xml_content.replace('type="hfield" hfield="terrain"', 'type="plane" size="30 30 0.1"')

        soccer_arena_xml = f"""
        <!-- RoboCup Soccer Goal Posts at X=4.0m -->
        <body name="goal_post_left" pos="4.0 -0.9 0.55">
            <geom name="geom_goal_post_left" type="cylinder" size="0.035 0.55" rgba="0.95 0.95 0.95 1.0"/>
        </body>
        <body name="goal_post_right" pos="4.0 0.9 0.55">
            <geom name="geom_goal_post_right" type="cylinder" size="0.035 0.55" rgba="0.95 0.95 0.95 1.0"/>
        </body>
        <body name="goal_crossbar" pos="4.0 0.0 1.10">
            <geom name="geom_goal_crossbar" type="cylinder" size="0.035 0.90" euler="1.5708 0 0" rgba="0.95 0.95 0.95 1.0"/>
        </body>
        <body name="goal_net_back" pos="4.45 0.0 0.55">
            <geom name="geom_net_back" type="box" size="0.015 0.90 0.55" rgba="0.85 0.85 0.85 0.30"/>
            <geom name="geom_net_left" type="box" size="0.45 0.015 0.55" pos="-0.225 -0.9 0" rgba="0.85 0.85 0.85 0.25"/>
            <geom name="geom_net_right" type="box" size="0.45 0.015 0.55" pos="-0.225 0.9 0" rgba="0.85 0.85 0.85 0.25"/>
            <geom name="geom_net_top" type="box" size="0.45 0.90 0.015" pos="-0.225 0 0.55" rgba="0.85 0.85 0.85 0.25"/>
        </body>

        <!-- Field Visual Lines -->
        <body name="field_lines" pos="2.0 0.0 0.002">
            <geom name="geom_goal_line" type="box" size="0.025 1.5 0.001" pos="2.0 0 0" rgba="0.95 0.95 0.95 0.8"/>
            <geom name="geom_penalty_box_top" type="box" size="0.75 0.025 0.001" pos="1.25 1.5 0" rgba="0.95 0.95 0.95 0.8"/>
            <geom name="geom_penalty_box_bottom" type="box" size="0.75 0.025 0.001" pos="1.25 -1.5 0" rgba="0.95 0.95 0.95 0.8"/>
            <geom name="geom_penalty_box_front" type="box" size="0.025 1.5 0.001" pos="0.5 0 0" rgba="0.95 0.95 0.95 0.8"/>
        </body>

        <!-- RoboCup Soccer Ball with Newton Freejoint -->
        <body name="soccer_ball" pos="{ball_pos[0]} {ball_pos[1]} 0.065">
            <freejoint name="ball_joint"/>
            <geom name="ball_geom" type="sphere" size="0.065" mass="0.08"
                  rgba="1.0 0.25 0.05 1.0" friction="0.60 0.005 0.0001"
                  solref="0.003 1.0" solimp="0.95 0.99 0.001"/>
        </body>
        """
        xml_arena = xml_flat.replace("</worldbody>", soccer_arena_xml + "\n</worldbody>")

        self.model = mujoco.MjModel.from_xml_string(xml_arena)
        self.data = mujoco.MjData(self.model)

        self.ball_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        self.ball_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qposadr = self.model.jnt_qposadr[self.ball_jnt_id]
        self.ball_dofadr = self.model.jnt_dofadr[self.ball_jnt_id]

        self.initial_ball_pos = np.array([ball_pos[0], ball_pos[1], 0.065], dtype=np.float32)

        # Base Pose
        self.standing_legs = np.array([
            0.0, 0.0, -0.20, 0.30, -0.10, 0.0,  # Left
            0.0, 0.0, -0.20, 0.30, -0.10, 0.0   # Right
        ], dtype=np.float32)
        self.standing_root_z = -0.0045

        # Upper body joints
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

        # Simulation & Controller state
        self.dt = 1.0 / 60.0
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
        self.kick_time = 0.0
        self.kick_duration = 0.65
        self.peak_ball_speed = 0.0

        # Autonomous Striker AI state
        self.autonomous_ai = True
        self.fsm_state = "APPROACH"  # APPROACH -> ALIGN -> KICK -> EVALUATE -> CELEBRATE
        self.celebrate_timer = 0.0
        self.score = 0
        self.goal_scored_current = False

        self.cam_mode = 0  # 0: Chase Cam, 1: Broadcast Cam, 2: Goal Cam
        self.reset_requested = False

        self.reset_sim()

    def reset_sim(self, new_ball_pos: tuple = None):
        """Thread-safe reset of robot, ball, and controller."""
        mujoco.mj_resetData(self.model, self.data)

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
        self.kick_time = 0.0
        self.peak_ball_speed = 0.0
        self.fsm_state = "APPROACH"
        self.celebrate_timer = 0.0
        self.goal_scored_current = False
        self.reset_requested = False

        # Reset robot base & joints
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.standing_root_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0

        for i, qadr in enumerate(LEG_QPOS_ADR):
            self.data.qpos[qadr] = self.standing_legs[i]

        for act_id, val in self.upper_act_dict.items():
            self.data.ctrl[act_id] = val

        self.data.qpos[9] = 1.57
        self.data.qpos[10] = -0.15
        self.data.qpos[11] = -0.80
        self.data.qpos[12] = 1.57
        self.data.qpos[13] = 0.15
        self.data.qpos[14] = -0.80

        # Respawn soccer ball
        if new_ball_pos is not None:
            self.initial_ball_pos[0] = new_ball_pos[0]
            self.initial_ball_pos[1] = new_ball_pos[1]

        self.data.qpos[self.ball_qposadr:self.ball_qposadr+3] = self.initial_ball_pos
        self.data.qpos[self.ball_qposadr+3] = 1.0
        self.data.qpos[self.ball_qposadr+4:self.ball_qposadr+7] = 0.0
        self.data.qvel[self.ball_dofadr:self.ball_dofadr+6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def trigger_kick(self):
        if not self.is_kicking:
            self.is_kicking = True
            self.kick_time = 0.0
            print("\n[ACTION] DYNAMIC SOCCER KICK EXECUTED!")

    def key_callback(self, keycode: int):
        speed_step = 0.18
        turn_step = 0.25

        if keycode in [KEY_T, 84, 116]:
            self.autonomous_ai = not self.autonomous_ai
            status = "AUTONOMOUS STRIKER AI ACTIVE" if self.autonomous_ai else "MANUAL TELEOPERATION"
            print(f"\n[MODE] Switched to: {status}")
        elif keycode in [KEY_R, 82, 114]:
            self.reset_requested = True
            print("\n[RESET REQUESTED]")
        elif keycode in [KEY_B, 66, 98]:
            # Respawn ball with small random jitter
            bx = 2.0 + np.random.uniform(-0.3, 0.3)
            by = 0.2 + np.random.uniform(-0.4, 0.4)
            self.reset_sim(new_ball_pos=(bx, by))
            print(f"\n[BALL RESPAWN] New ball position: ({bx:.2f}, {by:.2f})")
        elif keycode in [KEY_V, 86, 118]:
            self.cam_mode = (self.cam_mode + 1) % 3
            cam_names = ["Dynamic Chase Cam", "Broadcast Arena Cam", "Goal Net Cam"]
            print(f"\n[CAMERA] Switched to: {cam_names[self.cam_mode]}")
        elif not self.autonomous_ai:
            # Manual controls active only when AI is OFF
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
            elif keycode in [KEY_K, KEY_SPACE]:
                self.trigger_kick()
            elif keycode == KEY_1:
                self.target_vx = 0.50
            elif keycode == KEY_2:
                self.target_vx = 0.80
            elif keycode == KEY_3:
                self.target_vx = 1.55

    def compute_kick_pose(self, tau: float) -> np.ndarray:
        """Bio-mechanical human soccer kick trajectory."""
        legs = self.standing_legs.copy()
        if tau < 0.30:  # Windup: shift weight to left stance, draw right leg back
            s = tau / 0.30
            legs[1] -= 0.14 * s  # L hip roll (weight shift)
            legs[8] += 0.52 * s  # R hip pitch back
            legs[9] += 0.98 * s  # R knee bend
        elif tau < 0.60:  # Explosive forward strike
            s = (tau - 0.30) / 0.30
            legs[1] -= 0.14
            legs[8] = 0.52 - 1.42 * s   # R hip pitch forward to -0.90 rad
            legs[9] = 1.28 - 1.08 * s   # R knee snaps forward to 0.20 rad
            legs[10] = -0.10 - 0.25 * s # R ankle stiff
        else:  # Follow-through and recovery
            s = (tau - 0.60) / 0.40
            legs[1] = -0.14 * (1.0 - s)
            legs[8] = -0.90 * (1.0 - s) + (-0.20) * s
            legs[9] = 0.20 * (1.0 - s) + 0.30 * s
            legs[10] = -0.35 * (1.0 - s) + (-0.10) * s
        return legs

    def update_autonomous_ai(self):
        """Finite State Machine for Autonomous Striker AI."""
        ball_pos = self.data.qpos[self.ball_qposadr:self.ball_qposadr+3]
        ball_vel = self.data.qvel[self.ball_dofadr:self.ball_dofadr+3]
        ball_speed = math.sqrt(ball_vel[0]**2 + ball_vel[1]**2)

        goal_x, goal_y = 4.0, 0.0

        if self.fsm_state == "APPROACH":
            dx = ball_pos[0] - self.pos_x
            dy = ball_pos[1] - self.pos_y
            dist_to_ball = math.sqrt(dx**2 + dy**2)
            target_heading = math.atan2(dy, dx)
            angle_err = wrap_to_pi(target_heading - self.heading)

            if dist_to_ball > 0.80:
                self.target_wz = np.clip(2.5 * angle_err, -1.0, 1.0)
                if abs(angle_err) < 0.30:
                    self.target_vx = 1.45 if dist_to_ball > 1.35 else 0.80
                else:
                    self.target_vx = 0.30
                self.target_vy = 0.0
            else:
                self.fsm_state = "ALIGN"

        elif self.fsm_state == "ALIGN":
            theta_shot = math.atan2(goal_y - ball_pos[1], goal_x - ball_pos[0])
            # Right foot strike sweet-spot (0.180m forward, -0.068m lateral)
            dx_offset = 0.180 * math.cos(theta_shot) - (-0.068) * math.sin(theta_shot)
            dy_offset = 0.180 * math.sin(theta_shot) + (-0.068) * math.cos(theta_shot)
            r_x = ball_pos[0] - dx_offset
            r_y = ball_pos[1] - dy_offset

            ex = r_x - self.pos_x
            ey = r_y - self.pos_y
            e_dist = math.sqrt(ex**2 + ey**2)
            e_heading = wrap_to_pi(theta_shot - self.heading)

            vx_body = ex * math.cos(self.heading) + ey * math.sin(self.heading)
            vy_body = -ex * math.sin(self.heading) + ey * math.cos(self.heading)

            self.target_vx = np.clip(1.8 * vx_body, -0.20, 0.40)
            self.target_vy = np.clip(1.8 * vy_body, -0.25, 0.25)
            self.target_wz = np.clip(2.5 * e_heading, -0.80, 0.80)

            speed = math.sqrt(self.cmd_vx**2 + self.cmd_vy**2)
            if e_dist < 0.020 and abs(e_heading) < 0.06 and speed < 0.08:
                self.fsm_state = "KICK"
                self.target_vx = 0.0
                self.target_vy = 0.0
                self.target_wz = 0.0
                self.trigger_kick()

        elif self.fsm_state == "KICK":
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_wz = 0.0
            if not self.is_kicking:
                self.fsm_state = "EVALUATE"

        elif self.fsm_state == "EVALUATE":
            if ball_pos[0] >= 4.0 and abs(ball_pos[1]) <= 0.90:
                self.fsm_state = "CELEBRATE"
                self.celebrate_timer = 0.0
                if not self.goal_scored_current:
                    self.score += 1
                    self.goal_scored_current = True
                    print(f"\n[GOAL!] SCORE: {self.score} - 0! Ball in net at ({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")
            elif ball_speed < 0.05:
                # Missed or stopped short -> chase again
                self.fsm_state = "APPROACH"

        elif self.fsm_state == "CELEBRATE":
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_wz = 0.0
            self.celebrate_timer += self.dt
            if self.celebrate_timer > 3.5:
                # Auto reset after celebration
                bx = 2.0 + np.random.uniform(-0.2, 0.2)
                by = 0.2 + np.random.uniform(-0.3, 0.3)
                self.reset_sim(new_ball_pos=(bx, by))

    def physics_step(self):
        if self.reset_requested:
            self.reset_sim()
            print("[RESET] Simulation reset to start!")
            return

        # 1. Autonomous AI update
        if self.autonomous_ai:
            self.update_autonomous_ai()

        # 2. Smooth velocity commands
        dt = self.dt
        acc_lin = 3.5 * dt
        acc_ang = 4.5 * dt
        self.cmd_vx = np.clip(self.target_vx, self.cmd_vx - acc_lin, self.cmd_vx + acc_lin)
        self.cmd_vy = np.clip(self.target_vy, self.cmd_vy - acc_lin, self.cmd_vy + acc_lin)
        self.cmd_wz = np.clip(self.target_wz, self.cmd_wz - acc_ang, self.cmd_wz + acc_ang)

        speed = math.sqrt(self.cmd_vx**2 + self.cmd_vy**2)

        # 3. Blending state transitions
        if speed < 0.05 and abs(self.cmd_wz) < 0.05 and not self.is_kicking:
            self.stand_blend = min(1.0, self.stand_blend + 5.0 * dt)
        else:
            self.stand_blend = max(0.0, self.stand_blend - 5.0 * dt)

        target_run_b = np.clip((speed - 0.75) / (1.35 - 0.75), 0.0, 1.0)
        self.run_blend += np.clip(target_run_b - self.run_blend, -3.0 * dt, 3.0 * dt)

        # 4. Heading & World Position Integration
        prev_pos_x = self.pos_x
        prev_pos_y = self.pos_y
        self.heading += self.cmd_wz * dt
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        self.pos_x += (cos_h * self.cmd_vx - sin_h * self.cmd_vy) * dt
        self.pos_y += (sin_h * self.cmd_vx + cos_h * self.cmd_vy) * dt

        # 5. Gait Phase Propagation
        if self.stand_blend < 0.99 and not self.is_kicking:
            w_speed_ratio = max(0.2, min(1.1, speed / self.walk_speed_ref)) if speed > 0.05 else 0.5 * abs(self.cmd_wz)
            self.walk_phase += 60.0 * w_speed_ratio * dt
            if self.walk_phase >= self.walk_cycle_end:
                self.walk_phase = self.walk_cycle_start + (self.walk_phase - self.walk_cycle_end) % self.walk_cycle_len

            r_speed_ratio = max(0.6, speed / self.run_speed_ref)
            self.run_phase += 60.0 * r_speed_ratio * dt
            if self.run_phase >= self.run_cycle_end:
                self.run_phase = self.run_cycle_start + (self.run_phase - self.run_cycle_end) % self.run_cycle_len

        # 6. Pose Interpolation
        f_w0 = int(math.floor(self.walk_phase))
        f_w1 = (f_w0 + 1) if (f_w0 + 1) < self.walk_cycle_end else self.walk_cycle_start
        a_w = self.walk_phase - f_w0
        walk_legs = (1.0 - a_w) * self.walk_qpos[f_w0, LEG_QPOS_ADR] + a_w * self.walk_qpos[f_w1, LEG_QPOS_ADR]
        walk_root_z = (1.0 - a_w) * self.walk_qpos[f_w0, 2] + a_w * self.walk_qpos[f_w1, 2]

        f_r0 = int(math.floor(self.run_phase))
        f_r1 = (f_r0 + 1) if (f_r0 + 1) < self.run_cycle_end else self.run_cycle_start
        a_r = self.run_phase - f_r0
        run_legs = (1.0 - a_r) * self.run_qpos[f_r0, LEG_QPOS_ADR] + a_r * self.run_qpos[f_r1, LEG_QPOS_ADR]
        run_root_z = (1.0 - a_r) * self.run_qpos[f_r0, 2] + a_r * self.run_qpos[f_r1, 2]

        r_b = self.run_blend
        loco_legs = (1.0 - r_b) * walk_legs + r_b * run_legs
        loco_root_z = (1.0 - r_b) * walk_root_z + r_b * run_root_z

        # Dynamic Kick Trajectory
        if self.is_kicking:
            self.kick_time += dt
            if self.kick_time < self.kick_duration:
                tau = self.kick_time / self.kick_duration
                final_legs = self.compute_kick_pose(tau)
            else:
                self.is_kicking = False
                final_legs = self.standing_legs
            final_root_z = self.standing_root_z
        else:
            s_b = self.stand_blend
            final_legs = s_b * self.standing_legs + (1.0 - s_b) * loco_legs
            final_root_z = s_b * self.standing_root_z + (1.0 - s_b) * loco_root_z

        # 7. Apply to MuJoCo
        prev_qpos = self.data.qpos.copy()
        self.data.qpos[0] = self.pos_x
        self.data.qpos[1] = self.pos_y
        self.data.qpos[2] = final_root_z

        q_h = sptr.Rotation.from_euler("z", self.heading).as_quat()
        self.data.qpos[3:7] = [q_h[3], q_h[0], q_h[1], q_h[2]]

        for i, qadr in enumerate(LEG_QPOS_ADR):
            self.data.qpos[qadr] = final_legs[i]

        # Upper Body Pose
        if self.fsm_state == "CELEBRATE":
            # Arms raised in victory V-pose!
            self.data.qpos[9]  = 0.20   # R shoulder pitch up
            self.data.qpos[10] = -0.55  # R shoulder roll out
            self.data.qpos[11] = -0.40  # R elbow
            self.data.qpos[12] = 0.20   # L shoulder pitch up
            self.data.qpos[13] = 0.55   # L shoulder roll out
            self.data.qpos[14] = -0.40  # L elbow
        elif self.is_kicking:
            tau = self.kick_time / self.kick_duration
            arm_swing = 0.45 * math.sin(math.pi * tau)
            self.data.qpos[9]  = 1.57 - arm_swing
            self.data.qpos[12] = 1.57 + arm_swing
        else:
            arm_amp = 0.35 * (1.0 - r_b) + 0.65 * r_b
            arm_pitch_offset = -0.40 * r_b
            arm_swing = (1.0 - self.stand_blend) * arm_amp * math.sin(
                2.0 * math.pi * (self.run_phase if r_b > 0.5 else self.walk_phase) / 60.0
            )
            self.data.qpos[9]  = 1.57 - arm_swing + arm_pitch_offset
            self.data.qpos[10] = -0.15
            self.data.qpos[11] = -0.75 * (1.0 - r_b) - 1.25 * r_b
            self.data.qpos[12] = 1.57 + arm_swing + arm_pitch_offset
            self.data.qpos[13] = 0.15
            self.data.qpos[14] = -0.75 * (1.0 - r_b) - 1.25 * r_b

        # Compute dynamic velocities for Newton contact impulse
        self.data.qvel[0] = (self.pos_x - prev_qpos[0]) / dt
        self.data.qvel[1] = (self.pos_y - prev_qpos[1]) / dt
        self.data.qvel[2] = 0.0
        for i, qadr in enumerate(LEG_QPOS_ADR):
            dof = self.model.jnt_dofadr[self.model.jnt_qposadr.tolist().index(qadr)]
            self.data.qvel[dof] = (final_legs[i] - prev_qpos[qadr]) / dt

        # 8. True Newton Physics Integration for ball and contacts
        substeps = int(round(dt / self.model.opt.timestep))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        # Track ball velocity
        b_vx = self.data.qvel[self.ball_dofadr]
        b_vy = self.data.qvel[self.ball_dofadr + 1]
        b_v = math.sqrt(b_vx**2 + b_vy**2)
        self.peak_ball_speed = max(self.peak_ball_speed, b_v)

    def run_gui(self):
        try:
            import mujoco.viewer
        except ImportError:
            print("[ERROR] mujoco.viewer requires OpenGL/GLFW. Use --render-demo instead.")
            return

        print("=" * 82)
        print("  BARELANG-S AUTONOMOUS ROBOCUP STRIKER SIMULATION (SUPERMIMIC)")
        print("=" * 82)
        print("  Controls:")
        print("    [T]           : Toggle Autonomous Striker AI (RoboCup Mode)")
        print("    [W] / [S]     : Forward / Backward (Walk -> Sprint)")
        print("    [A] / [D]     : Turn Left / Right")
        print("    [Q] / [E]     : Strafe Left / Right")
        print("    [K] / [SPACE] : Dynamic Soccer Kick")
        print("    [SPACE]       : Stop / Stand Still")
        print("    [B]           : Respawn Soccer Ball")
        print("    [R]           : Reset Robot & Ball to Origin")
        print("    [V]           : Cycle Camera Views (Chase -> Broadcast -> Goal Cam)")
        print("=" * 82)

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
            last_hud = 0.0

            while viewer.is_running():
                step_start = time.time()
                self.physics_step()

                # Camera views
                if self.cam_mode == 0:  # Dynamic Chase Cam
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = self.torso_bid
                    viewer.cam.distance = 2.4
                    viewer.cam.elevation = -14.0
                    viewer.cam.azimuth = math.degrees(self.heading) + 90.0
                elif self.cam_mode == 1:  # TV Broadcast Cam
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat = [2.2, 0.0, 0.40]
                    viewer.cam.distance = 5.2
                    viewer.cam.elevation = -24.0
                    viewer.cam.azimuth = 90.0
                else:  # Goal Cam
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat = [2.0, 0.0, 0.40]
                    viewer.cam.distance = 4.2
                    viewer.cam.elevation = -10.0
                    viewer.cam.azimuth = -90.0

                viewer.sync()

                now = time.time()
                if now - last_hud > 0.12:
                    last_hud = now
                    ball_x = self.data.qpos[self.ball_qposadr]
                    ball_y = self.data.qpos[self.ball_qposadr + 1]
                    dist_to_ball = math.sqrt((ball_x - self.pos_x)**2 + (ball_y - self.pos_y)**2)

                    mode_str = f"AUTO-STRIKER [{self.fsm_state}]" if self.autonomous_ai else "MANUAL TELEOP"
                    hud = (
                        f"\r[{mode_str:24s}] SCORE: {self.score}-0 | Ball: ({ball_x:.2f}, {ball_y:.2f}) "
                        f"Dist: {dist_to_ball:.2f}m | Peak V: {self.peak_ball_speed*3.6:4.1f}km/h | Speed: {self.cmd_vx:.2f}m/s  "
                    )
                    sys.stdout.write(hud)
                    sys.stdout.flush()

                elapsed = time.time() - step_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

    def render_demo(self, output_path: str = "data/mocap/barelang_soccer_striker_match.mp4", duration_sec: float = 6.0):
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) required for video rendering.")
            return

        print(f"\nRendering Autonomous RoboCup Striker Match ({duration_sec}s @ 60fps) to: {output_path}")
        renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat = [2.1, 0.0, 0.35]
        camera.distance = 4.8
        camera.elevation = -22.0
        camera.azimuth = 80.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, 60.0, (640, 480))

        self.reset_sim(new_ball_pos=(2.0, 0.15))
        self.autonomous_ai = True

        total_frames = int(duration_sec * 60)

        for f in range(total_frames):
            sim_time = f * self.dt
            self.physics_step()

            # Dynamic camera zoom and smooth tracking as robot approaches ball and kicks
            if sim_time < 3.2:
                # Approach phase: follow midfield
                camera.lookat = [1.5 + 0.3 * sim_time, 0.10, 0.35]
                camera.distance = 4.8 - 0.25 * sim_time
            else:
                # Kick & Goal phase: focus on goal and ball flight
                ball_x = self.data.qpos[self.ball_qposadr]
                ball_y = self.data.qpos[self.ball_qposadr + 1]
                camera.lookat = [min(3.8, max(2.5, ball_x)), ball_y * 0.5, 0.35]
                camera.distance = 4.0

            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()

            ball_x = self.data.qpos[self.ball_qposadr]
            ball_y = self.data.qpos[self.ball_qposadr + 1]
            dist_to_ball = math.sqrt((ball_x - self.pos_x)**2 + (ball_y - self.pos_y)**2)

            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Draw HUD
            hud_title = f"BARELANG-S ROBOCUP SOCCER STRIKER AI  |  SCORE: {self.score} - 0"
            hud_status = f"T={sim_time:4.2f}s | State: {self.fsm_state:10s} | Speed: {self.cmd_vx:.2f} m/s | Ball V: {self.peak_ball_speed*3.6:4.1f} km/h"

            cv2.rectangle(frame, (10, 10), (630, 68), (20, 20, 20), -1)
            cv2.putText(frame, hud_title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.putText(frame, hud_status, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2)

            if self.fsm_state == "CELEBRATE":
                cv2.putText(frame, "GOAL! 1 - 0!", (220, 180), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

            writer.write(frame)

            if f % 30 == 0 or f == total_frames - 1:
                print(f"\rRendered {f+1}/{total_frames} frames ({f/total_frames*100:.1f}%) | "
                      f"State: {self.fsm_state:10s} | Score: {self.score}-0 | Peak V: {self.peak_ball_speed*3.6:.1f} km/h", end="", flush=True)

        writer.release()
        print(f"\n[OK] Autonomous Striker match video successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BarelangS Autonomous RoboCup Striker AI")
    parser.add_argument("--walk", type=str, default="data/mocap/barelang_mocap_walk.npz")
    parser.add_argument("--run", type=str, default="data/mocap/barelang_mocap_run.npz")
    parser.add_argument("--xml", type=str, default="barelangS_terrain_mujoco.xml")
    parser.add_argument("--render-demo", type=str, default=None, help="Path to render demo MP4")
    parser.add_argument("--duration", type=float, default=6.0, help="Demo duration in seconds")
    args = parser.parse_args()

    sim = SoccerStrikerSimulation(
        walk_path=args.walk,
        run_path=args.run,
        xml_base_path=args.xml
    )

    if args.render_demo:
        sim.render_demo(output_path=args.render_demo, duration_sec=args.duration)
    else:
        sim.run_gui()
