import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class BarelangSWalkEnv(gym.Env):
    """
    Gymnasium RL Environment for BarelangS Humanoid Bipedal Locomotion.
    Features:
      - Omnidirectional Residual CPG: Forward, Backward, Strafe Left/Right, and Turn.
      - Active toe clearance (dorsiflexion during swing) eliminating foot snagging / "nyangkut".
      - Symmetrical multi-directional velocity tracking rewards.
      - Push-disturbance rejection training for rock-solid stability.
      - Dynamixel MX-106 (legs), MX-64 (hip yaw & arms), MX-28 (head).
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path=None, training=True, frame_skip=10):
        super().__init__()
        
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "barelangS_mujoco.xml")
            
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML model not found at: {xml_path}")
            
        self.xml_path = xml_path
        self.training = training
        self.frame_skip = frame_skip  # 10 * 0.002s = 0.02s (50 Hz control frequency)
        self.dt = 0.002 * frame_skip # 0.02s
        self.gait_frequency = 1.45   # 1.45 Hz stepping frequency

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # 12 Controlled Leg Joints in order
        self.leg_joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll"
        ]
        self.num_leg_joints = len(self.leg_joint_names)

        # Map leg joint names to actuator indices and qpos/qvel addresses
        self.leg_act_indices = []
        self.leg_qpos_indices = []
        self.leg_qvel_indices = []
        for name in self.leg_joint_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.leg_act_indices.append(act_id)
            self.leg_qpos_indices.append(self.model.jnt_qposadr[jnt_id])
            self.leg_qvel_indices.append(self.model.jnt_dofadr[jnt_id])

        # Passive / stabilized upper body posture
        self.upper_body_posture = {
            "right_shoulder_pitch": 1.57,
            "left_shoulder_pitch": 1.57,
            "right_shoulder_roll": -0.15,
            "left_shoulder_roll": 0.15,
            "right_elbow": -0.80,
            "left_elbow": -0.80,
            "head_yaw": 0.0,
            "head_pitch": -0.30
        }
        self.upper_act_dict = {}
        for name, target in self.upper_body_posture.items():
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.upper_act_dict[act_id] = (qpos_adr, target)

        # Residual action space: normalized offsets in [-1, 1] scaled by 0.14 rad (~8.0 deg)
        self.action_scale = 0.14
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_leg_joints,), dtype=np.float32
        )

        # Observation space (54 dimensions: including heading error sin/cos)
        obs_dim = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 2 + 2 # 54
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Key site & body identifiers
        self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torso")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.r_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll")
        self.l_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Initial state variables
        self.initial_root_z = -0.005
        self.nominal_torso_height = 0.420
        self.max_episode_steps = 1000 # 20 seconds at 50 Hz
        self.command = np.zeros(3, dtype=np.float32) # [vx, vy, wz]
        self.target_v = np.zeros(3, dtype=np.float32) # ramped command
        self.target_yaw = 0.0
        self.prev_action = np.zeros(self.num_leg_joints, dtype=np.float32)
        self.phase = 0.0
        self.step_counter = 0
        self.command_duration = 0

        # Push disturbance variables
        self.push_steps = 0
        self.push_force = np.zeros(3, dtype=np.float32)

    def compute_cpg(self, phi: float, vx: float, vy: float, wz: float) -> np.ndarray:
        """
        Computes dynamic bipedal walking trajectory for all 12 leg joints.
        Order: [lhy, lhr, lhp, lk, lap, lar, rhy, rhr, rhp, rk, rap, rar]
        Features:
          - Balanced CoM equilibrium (nom_pitch = -0.26, lk = 0.52, lap = -0.26)
          - Anti-forward-fall & anti-backward-fall bounded stride (step_p <= 0.10)
          - Corrected dynamic sway towards STANCE foot (unloading swing foot)
          - Symmetrical & maximized lateral strafe (left & right up to 0.20 rad)
          - Differential pitch turning couple for brisk in-place yaw rotation (no heavy rear)
        """
        speed = math.sqrt(vx*vx + vy*vy) + abs(wz) * 0.1
        if speed < 0.02:
            return np.array([
                0.0,   0.015, -0.26,  0.52, -0.26, -0.015,  # Left leg
                0.0,  -0.015, -0.26,  0.52, -0.26,  0.015   # Right leg
            ], dtype=np.float32)

        sin_p = math.sin(phi)
        swing_l = max(0.0, sin_p)
        swing_r = max(0.0, -sin_p)

        # 1. Pitch & Knee Kinematics: Balanced Torso Pitch & Anti-Falling Margins
        if vx >= 0.0:
            nom_pitch = -0.26 - 0.03 * math.tanh(3.0 * vx)
            step_p = np.clip(0.35 * vx, 0.0, 0.10)
            clearance_pitch = -0.06 * math.tanh(3.0 * vx)
        else:
            # Lean slightly forward (-0.28 rad) during backward walking to maintain 9cm heel margin
            nom_pitch = -0.26 - 0.02 * abs(math.tanh(3.0 * vx))
            step_p = np.clip(0.50 * vx, -0.13, 0.0) # Active backward stride
            clearance_pitch = 0.0 # Flat foot during backward swing gives maximum floor clearance (4.4 cm)

        turn_pitch = np.clip(0.12 * wz, -0.07, 0.07)
        knee_nom = 0.52 + 0.02 * math.tanh(3.0 * vx)
        
        # High knee lift for generous foot clearance (> 5 cm)
        knee_a = np.clip(0.44 + 0.16 * abs(vx) + 0.28 * abs(vy), 0.42, 0.54)

        l_toe_lift = clearance_pitch * swing_l
        r_toe_lift = clearance_pitch * swing_r

        # Stance knee extends slightly, swing knee flexes strongly
        lk = (knee_nom - 0.06 * swing_r) + knee_a * swing_l
        rk = (knee_nom - 0.06 * swing_l) + knee_a * swing_r

        # Hip pitch with differential yaw turn couple
        lhp = nom_pitch - step_p * sin_p + turn_pitch * sin_p
        rhp = nom_pitch + step_p * sin_p - turn_pitch * sin_p

        # Exact Geometric Ankle Pitch (Zero angle error relative to ground + active clearance)
        lap = -(lhp + lk) + l_toe_lift
        rap = -(rhp + rk) + r_toe_lift

        # 2. Lateral Sway towards STANCE Foot (Sway is negative when left foot swings)
        sway = -(0.048 + 0.012 * abs(vx) + 0.025 * abs(vy)) * sin_p
        if vy >= 0.0: # Strafe Left
            lat_left  = np.clip( 1.35 * vy, 0.0, 0.20) * swing_l
            lat_right = np.clip( 0.45 * vy, 0.0, 0.05) * swing_r
        else: # Strafe Right
            lat_left  = np.clip( 0.45 * vy, -0.05, 0.0) * swing_l
            lat_right = np.clip( 1.35 * vy, -0.20, 0.0) * swing_r

        # Safe Hip Roll (Stay strictly within physical joint limits)
        lhr = np.clip( 0.015 + sway + lat_left,  -0.08, 0.28)
        rhr = np.clip(-0.015 + sway + lat_right, -0.28, 0.08)
        lar = -lhr
        rar = -rhr

        # 3. Pure In-Place Yaw Rotation Stepping
        yaw_step = np.clip(0.32 * wz, -0.20, 0.20)
        lhy = yaw_step * swing_l
        rhy = yaw_step * swing_r

        return np.array([lhy, lhr, lhp, lk, lap, lar, rhy, rhr, rhp, rk, rap, rar], dtype=np.float32)

    def set_command(self, vx: float, vy: float, wz: float):
        """Allows direct external control of velocity commands (for Sim2Sim keyboard interaction)."""
        self.command = np.array([vx, vy, wz], dtype=np.float32)

    def _sample_command(self):
        """Domain randomization of velocity commands during RL training (balanced across modes)."""
        mode = np.random.choice([
            "forward", "backward", "strafe_left", "strafe_right", 
            "turn_in_place_ccw", "turn_in_place_cw", "combined", "stop"
        ], p=[0.20, 0.25, 0.13, 0.16, 0.11, 0.11, 0.03, 0.01])
        if mode == "forward":
            self.command = np.array([np.random.uniform(0.12, 0.25), 0.0, 0.0], dtype=np.float32)
        elif mode == "backward":
            self.command = np.array([np.random.uniform(-0.15, -0.06), 0.0, 0.0], dtype=np.float32)
        elif mode == "strafe_left":
            self.command = np.array([0.0, np.random.uniform(0.06, 0.16), 0.0], dtype=np.float32)
        elif mode == "strafe_right":
            self.command = np.array([0.0, np.random.uniform(-0.16, -0.06), 0.0], dtype=np.float32)
        elif mode == "turn_in_place_ccw":
            self.command = np.array([0.0, 0.0, np.random.uniform(0.15, 0.45)], dtype=np.float32)
        elif mode == "turn_in_place_cw":
            self.command = np.array([0.0, 0.0, np.random.uniform(-0.45, -0.15)], dtype=np.float32)
        elif mode == "combined":
            fwd_or_bwd = np.random.choice([1.0, -1.0], p=[0.5, 0.5])
            vx = np.random.uniform(0.10, 0.20) if fwd_or_bwd > 0 else np.random.uniform(-0.14, -0.06)
            self.command = np.array([
                vx,
                np.random.uniform(-0.10, 0.10),
                np.random.uniform(-0.25, 0.25)
            ], dtype=np.float32)
        else: # stop
            self.command = np.zeros(3, dtype=np.float32)
            
        self.command_duration = np.random.randint(150, 300)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.phase = np.random.uniform(0.0, 2.0 * math.pi) if self.training else 0.0
        self.step_counter = 0
        self.push_steps = 0
        self.target_yaw = 0.0
        if self.training:
            self._sample_command()
            self.target_v = self.command.copy()
        else:
            self.target_v = np.zeros(3, dtype=np.float32)

        # Compute initial CPG posture
        init_cpg = self.compute_cpg(self.phase, *self.target_v)

        for i, qpos_adr in enumerate(self.leg_qpos_indices):
            noise = np.random.uniform(-0.015, 0.015) if self.training else 0.0
            self.data.qpos[qpos_adr] = init_cpg[i] + noise

        # Set upper body posture
        for act_id, (qpos_adr, val) in self.upper_act_dict.items():
            self.data.qpos[qpos_adr] = val
            self.data.ctrl[act_id] = val

        # Set root position and orientation
        self.data.qpos[0] = 0.0 # x
        self.data.qpos[1] = 0.0 # y
        self.data.qpos[2] = self.initial_root_z # z
        self.data.qpos[3] = 1.0 # quat w
        self.data.qpos[4] = 0.0 # quat x
        self.data.qpos[5] = 0.0 # quat y
        self.data.qpos[6] = 0.0 # quat z

        self.data.qvel[:] = 0.0

        for i, act_id in enumerate(self.leg_act_indices):
            self.data.ctrl[act_id] = self.data.qpos[self.leg_qpos_indices[i]]

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.num_leg_joints, dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        
        projected_gravity = -R[2, :]
        world_angvel = self.data.qvel[3:6]
        local_angvel = R.T @ world_angvel
        world_linvel = self.data.qvel[0:3]
        local_linvel = R.T @ world_linvel

        cpg_target = self.compute_cpg(self.phase, *self.target_v)
        curr_joint_pos = np.array([self.data.qpos[idx] for idx in self.leg_qpos_indices], dtype=np.float32)
        rel_joint_pos = curr_joint_pos - cpg_target
        joint_vel = np.array([self.data.qvel[idx] for idx in self.leg_qvel_indices], dtype=np.float32)

        sin_phase = math.sin(self.phase)
        cos_phase = math.cos(self.phase)
        r_contact, l_contact, _ = self._check_contacts()

        current_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        if abs(self.target_v[2]) < 0.02:
            yaw_diff = (current_yaw - self.target_yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            yaw_diff = 0.0
            self.target_yaw = current_yaw

        heading_feat = np.array([math.sin(yaw_diff), math.cos(yaw_diff)], dtype=np.float32)

        obs = np.concatenate([
            self.target_v,                                               # 3
            projected_gravity,                                           # 3
            local_angvel,                                                # 3
            local_linvel,                                                # 3
            rel_joint_pos,                                               # 12
            joint_vel * 0.1,                                             # 12
            self.prev_action,                                            # 12
            np.array([sin_phase, cos_phase], dtype=np.float32),          # 2
            np.array([float(l_contact), float(r_contact)], dtype=np.float32), # 2
            heading_feat                                                 # 2
        ]).astype(np.float32)

        return obs

    def _check_contacts(self):
        r_contact = False
        l_contact = False
        has_non_foot_contact = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.floor_geom_id or c.geom2 == self.floor_geom_id:
                other_geom = c.geom2 if c.geom1 == self.floor_geom_id else c.geom1
                bid = self.model.geom_bodyid[other_geom]
                if bid == self.r_foot_bid:
                    r_contact = True
                elif bid == self.l_foot_bid:
                    l_contact = True
                else:
                    has_non_foot_contact = True

        return r_contact, l_contact, has_non_foot_contact

    def step(self, action):
        self.step_counter += 1
        action = np.clip(action, -1.0, 1.0)

        if self.training:
            self.command_duration -= 1
            if self.command_duration <= 0:
                self._sample_command()

            # Random push disturbances during training (0.5% chance)
            if np.random.rand() < 0.005 and self.push_steps == 0:
                self.push_steps = np.random.randint(5, 12)
                self.push_force[:2] = np.random.uniform(-2.5, 2.5, size=2)

        # Smooth velocity command slew rate (0.5 m/s^2 max acceleration)
        max_dv = 0.5 * self.dt
        dv = np.clip(self.command - self.target_v, -max_dv, max_dv)
        self.target_v += dv

        # Advance CPG phase
        cmd_speed = math.sqrt(self.target_v[0]**2 + self.target_v[1]**2) + abs(self.target_v[2]) * 0.1
        if cmd_speed > 0.02:
            self.phase = (self.phase + 2.0 * math.pi * self.gait_frequency * self.dt) % (2.0 * math.pi)

        # Compute targets
        cpg_targets = self.compute_cpg(self.phase, *self.target_v)
        final_targets = cpg_targets + self.action_scale * action

        for i, act_id in enumerate(self.leg_act_indices):
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(final_targets[i], ctrl_min, ctrl_max)

        for act_id, (_, target) in self.upper_act_dict.items():
            self.data.ctrl[act_id] = target

        # Apply push force if active
        if self.push_steps > 0:
            self.data.xfrc_applied[self.torso_body_id, :2] = self.push_force[:2]
            self.push_steps -= 1
        else:
            self.data.xfrc_applied[self.torso_body_id, :2] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = bool(self.step_counter >= self.max_episode_steps)

        self.prev_action = action.copy()
        obs = self._get_obs()

        info = {
            "cmd_vx": self.target_v[0],
            "cmd_vy": self.target_v[1],
            "cmd_wz": self.target_v[2],
            "act_vx": self.data.qvel[0],
            "act_vy": self.data.qvel[1],
            "act_wz": self.data.qvel[5],
            "torso_z": self.data.site_xpos[self.torso_site_id, 2]
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        local_linvel = R.T @ self.data.qvel[0:3]
        local_angvel = R.T @ self.data.qvel[3:6]
        projected_gravity = -R[2, :]

        # 1. Forward / Backward Velocity Tracking (Decoupled X)
        err_vx = local_linvel[0] - self.target_v[0]
        r_vx = 15.0 * math.exp(-err_vx**2 / 0.010)
        r_fwd = 0.0
        p_stuck_x = 0.0
        if self.target_v[0] > 0.03: # Forward
            if local_linvel[0] > 0.02:
                r_fwd = 25.0 * min(local_linvel[0], self.target_v[0])
            else:
                p_stuck_x = 25.0 * (self.target_v[0] - max(0.0, local_linvel[0]))
        elif self.target_v[0] < -0.03: # Backward
            if local_linvel[0] < -0.02:
                r_fwd = 40.0 * min(-local_linvel[0], -self.target_v[0]) # High reward for real backward movement!
            else:
                p_stuck_x = 40.0 * (-self.target_v[0] - max(-0.02, local_linvel[0])) # Penalize stepping in place!

        # 2. Lateral Strafe Velocity Tracking (Decoupled Y)
        err_vy = local_linvel[1] - self.target_v[1]
        r_vy = 15.0 * math.exp(-err_vy**2 / 0.010)
        r_lat = 0.0
        p_stuck_y = 0.0
        if self.target_v[1] > 0.03: # Strafe Left
            if local_linvel[1] > 0.02:
                r_lat = 25.0 * min(local_linvel[1], self.target_v[1])
            else:
                p_stuck_y = 20.0 * (self.target_v[1] - max(0.0, local_linvel[1]))
        elif self.target_v[1] < -0.03: # Strafe Right
            if local_linvel[1] < -0.02:
                r_lat = 35.0 * min(-local_linvel[1], -self.target_v[1]) # High reward for real rightward movement!
            else:
                p_stuck_y = 30.0 * (-self.target_v[1] - max(-0.02, -local_linvel[1])) # Penalize stepping in place!

        # 3. Yaw Angular Velocity Tracking (Decoupled Yaw)
        err_wz = local_angvel[2] - self.target_v[2]
        r_wz = 15.0 * math.exp(-err_wz**2 / 0.015)
        r_rot = 0.0
        p_stuck_wz = 0.0
        if abs(self.target_v[2]) > 0.08:
            if abs(local_angvel[2]) > 0.04 and (local_angvel[2] * self.target_v[2] > 0):
                r_rot = 25.0 * min(abs(local_angvel[2]), abs(self.target_v[2]))
            else:
                p_stuck_wz = 20.0 * abs(self.target_v[2])

        # 4. STRICT ZERO-DRIFT PENALTIES (Anti-Melenceng)
        # When moving straight forward/backward: harshly penalize lateral drift Vy!
        p_lat_drift = 0.0
        if abs(self.target_v[1]) < 0.02:
            p_lat_drift = 40.0 * (local_linvel[1]**2)

        # When strafing: harshly penalize forward/backward drift Vx!
        p_fwd_drift = 0.0
        if abs(self.target_v[0]) < 0.02:
            p_fwd_drift = 40.0 * (local_linvel[0]**2)

        # When moving straight (wz == 0): harshly penalize turning/twisting rate!
        p_yaw_drift = 0.0
        if abs(self.target_v[2]) < 0.02:
            p_yaw_drift = 35.0 * (local_angvel[2]**2)

        # 5. HEADING LOCK (Autopilot Straight-Line Orientation Lock)
        current_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        r_heading = 0.0
        p_heading = 0.0
        if abs(self.target_v[2]) < 0.02:
            yaw_diff = (current_yaw - self.target_yaw + math.pi) % (2 * math.pi) - math.pi
            r_heading = 8.0 * math.exp(-yaw_diff**2 / 0.02)
            p_heading = 30.0 * (yaw_diff**2)
        else:
            self.target_yaw = current_yaw

        # 6. In-Place Pure Turn Linear Drift Penalty
        p_turn_drift = 0.0
        if abs(self.target_v[2]) > 0.08 and abs(self.target_v[0]) < 0.03 and abs(self.target_v[1]) < 0.03:
            p_turn_drift = 35.0 * (local_linvel[0]**2 + local_linvel[1]**2)

        # 7. Torso Upright Posture & Height Stability (Sag Prevention)
        r_upright = 8.0 * math.exp(-(1.0 + projected_gravity[2])**2 / 0.005)
        p_tilt = 16.0 * (projected_gravity[0]**2 + projected_gravity[1]**2)

        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        r_height = 8.0 * math.exp(-((torso_z - self.nominal_torso_height)**2) / 0.005)
        p_sag = 25.0 * (max(0.0, 0.36 - torso_z)**2)

        # 8. High Foot Clearance Reward & Ground Slip Penalty
        r_contact, l_contact, _ = self._check_contacts()
        sin_phase = math.sin(self.phase)
        lf_z = self.data.site_xpos[self.left_foot_site_id, 2]
        rf_z = self.data.site_xpos[self.right_foot_site_id, 2]

        r_clearance = 0.0
        cmd_speed = math.sqrt(self.target_v[0]**2 + self.target_v[1]**2)
        if cmd_speed > 0.03:
            if sin_phase > 0.35: # Left foot swing
                r_clearance += 4.0 * min(max(0.0, (lf_z - 0.008) / 0.040), 1.0)
            elif sin_phase < -0.35: # Right foot swing
                r_clearance += 4.0 * min(max(0.0, (rf_z - 0.008) / 0.040), 1.0)

        # Foot Slip Penalty (Penalizes foot dragging/shuffling while bearing weight)
        lf_vel = np.zeros(6)
        rf_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.left_foot_site_id, lf_vel, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.right_foot_site_id, rf_vel, 0)
        lf_vxy2 = lf_vel[3]**2 + lf_vel[4]**2
        rf_vxy2 = rf_vel[3]**2 + rf_vel[4]**2
        p_slip = 0.4 * (lf_vxy2 * float(l_contact) + rf_vxy2 * float(r_contact))

        # 9. Smoothness & energy penalties
        p_smoothness = 0.02 * np.sum(np.square(action - self.prev_action))
        p_torque = 1e-4 * np.sum(np.square(self.data.qfrc_actuator[self.leg_qvel_indices]))
        p_gyro = 0.2 * np.sum(np.square(local_angvel))

        # 10. Healthy Survival Bonus
        r_healthy = 4.0

        total_reward = (
            r_vx + r_vy + r_wz +
            r_fwd + r_lat + r_rot +
            r_heading +
            r_upright + r_height +
            r_clearance +
            r_healthy -
            p_stuck_x -
            p_stuck_y -
            p_stuck_wz -
            p_lat_drift -
            p_fwd_drift -
            p_yaw_drift -
            p_heading -
            p_turn_drift -
            p_tilt -
            p_gyro -
            p_smoothness -
            p_torque -
            p_slip -
            p_sag
        )

        return total_reward

    def _is_terminated(self):
        # 1. Fall detection by height (true collapse only: < 0.20m)
        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        if torso_z < 0.20 or torso_z > 0.55:
            return True

        # 2. Fall detection by excessive tilt (> 60 deg)
        q = self.data.qpos[3:7]
        w, x, y, z = q
        gz = -(1.0 - 2.0*(x**2 + y**2))
        if gz > -0.50:
            return True

        return False
