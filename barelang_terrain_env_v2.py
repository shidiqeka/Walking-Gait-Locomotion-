"""
Barelang S Terrain Environment v2
=================================
Complete rewrite focused on:
1. IMU-based torso correction & active balance recovery
2. Strong push disturbance training for robustness
3. Natural human-like walking from scratch (no transfer)
4. Larger action authority for policy to learn corrective movements
5. Curriculum learning: difficulty increases as policy improves
"""
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class BarelangSTerrainEnvV2(gym.Env):
    """
    MuJoCo Terrain Locomotion Environment for Barelang S Humanoid Robot.
    
    Key differences from v1:
    - Expanded observation space (62-dim) includes IMU tilt error signal
    - Larger action_scale (0.25 rad) gives policy more corrective authority
    - Aggressive push disturbance schedule trains push recovery
    - Tilt recovery reward: actively rewards returning to upright after perturbation
    - No velocity gating - policy must learn to stand still on its own
    - Curriculum: terrain difficulty auto-scales with episode survival rate
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self,
                 model_path="barelangS_terrain_mujoco.xml",
                 training=True,
                 terrain_type="random",
                 terrain_difficulty=0.3,
                 enable_curriculum=True):
        super().__init__()
        self.training = training
        self.terrain_type = terrain_type
        self.terrain_difficulty = terrain_difficulty
        self.enable_curriculum = enable_curriculum
        self.dt = 0.02  # 50 Hz control loop
        self.frame_skip = 10  # 500 Hz internal physics
        self.max_episode_steps = 800  # 16.0 seconds max per episode

        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Heightfield metadata
        self.hfield_nrow = self.model.hfield_nrow[0]
        self.hfield_ncol = self.model.hfield_ncol[0]
        self.hfield_size_x = self.model.hfield_size[0, 0]
        self.hfield_size_y = self.model.hfield_size[0, 1]
        self.hfield_max_z = self.model.hfield_size[0, 2]

        # Joint name definitions (12 leg DOF)
        self.leg_joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle_pitch", "right_ankle_roll"
        ]
        self.num_leg_joints = len(self.leg_joint_names)
        self.leg_qpos_indices = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.leg_joint_names
        ]
        self.leg_qvel_indices = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.leg_joint_names
        ]
        self.leg_act_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.leg_joint_names
        ]

        # Passive upper body posture (arms folded, head down)
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
        self.r_shoulder_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch")
        self.l_shoulder_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch")

        # ============================================================
        # KEY CHANGE: Larger action scale for corrective authority
        # v1 used 0.14 which limited the policy's ability to recover
        # v2 uses 0.25 to give policy more freedom for balance correction
        # ============================================================
        self.action_scale = 0.08

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_leg_joints,), dtype=np.float32
        )

        # ============================================================
        # EXPANDED OBSERVATION SPACE: 62 dimensions
        # Added: torso tilt error (2), tilt angular velocity (2), 
        #        terrain slope under feet (2), push indicator (2)
        # ============================================================
        # target_v(3) + gravity(3) + angvel(3) + linvel(3) + 
        # rel_joint_pos(12) + joint_vel(12) + prev_action(12) +
        # phase(2) + contacts(2) + heading(2) +
        # tilt_error(2) + tilt_rate(2) + terrain_slope(2) + push_indicator(2)
        obs_dim = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 2 + 2 + 2 + 2 + 2 + 2  # 62
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Key body & site identifiers
        self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torso")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.r_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll")
        self.l_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # CPG & kinematic constants
        self.gait_frequency = 1.4  # Hz (calm natural humanoid cadence)
        self.nominal_torso_height = 0.418
        self.initial_root_z = 0.0

        # State tracking
        self.command = np.zeros(3, dtype=np.float32)
        self.target_v = np.zeros(3, dtype=np.float32)
        self.target_yaw = 0.0
        self.prev_action = np.zeros(self.num_leg_joints, dtype=np.float32)
        self.phase = 0.0
        self.step_counter = 0
        self.command_duration = 0
        self.filtered_p_rate = 0.0
        self.filtered_r_rate = 0.0
        self.prev_ctrl_targets = None

        # ============================================================
        # PUSH RECOVERY STATE
        # Much stronger and more frequent pushes than v1
        # ============================================================
        self.push_steps = 0
        self.push_force = np.zeros(3, dtype=np.float32)
        self.push_active = False  # observable by policy
        self.prev_tilt_magnitude = 0.0  # for tilt recovery reward

        # Curriculum tracking
        self.episode_count = 0
        self.recent_survivals = []  # rolling window
        self.curriculum_difficulty = terrain_difficulty

        # Generate initial terrain
        self.generate_terrain(mode=self.terrain_type, difficulty=self.terrain_difficulty)

    # ================================================================
    # TERRAIN GENERATION
    # ================================================================
    def get_terrain_height(self, x: float, y: float) -> float:
        """Bilinear interpolated terrain elevation at world (x, y)."""
        u = np.clip((x + self.hfield_size_x) / (2.0 * self.hfield_size_x), 0.0, 1.0) * (self.hfield_ncol - 1)
        v = np.clip((y + self.hfield_size_y) / (2.0 * self.hfield_size_y), 0.0, 1.0) * (self.hfield_nrow - 1)

        col0 = int(np.floor(u))
        col1 = min(col0 + 1, self.hfield_ncol - 1)
        row0 = int(np.floor(v))
        row1 = min(row0 + 1, self.hfield_nrow - 1)
        du = u - col0
        dv = v - row0

        grid = np.array(self.model.hfield_data, dtype=np.float32).reshape((self.hfield_nrow, self.hfield_ncol))
        z00 = grid[row0, col0]
        z10 = grid[row0, col1]
        z01 = grid[row1, col0]
        z11 = grid[row1, col1]
        z_norm = (1 - du) * (1 - dv) * z00 + du * (1 - dv) * z10 + (1 - du) * dv * z01 + du * dv * z11
        return float(z_norm * self.hfield_max_z)

    def get_terrain_slope(self, x: float, y: float) -> tuple:
        """Returns terrain slope (dz/dx, dz/dy) at world (x, y)."""
        delta = 0.05  # 5cm finite difference
        zx_plus = self.get_terrain_height(x + delta, y)
        zx_minus = self.get_terrain_height(x - delta, y)
        zy_plus = self.get_terrain_height(x, y + delta)
        zy_minus = self.get_terrain_height(x, y - delta)
        return ((zx_plus - zx_minus) / (2 * delta), (zy_plus - zy_minus) / (2 * delta))

    def generate_terrain(self, mode="random", difficulty=0.6, flat_radius=0.25, center_x=0.0, center_y=0.0):
        """Generates procedural 3D terrain into MuJoCo hfield_data."""
        if mode == "random":
            mode = np.random.choice(
                ["rough", "wavy", "slopes", "mixed", "flat"],
                p=[0.35, 0.25, 0.20, 0.15, 0.05]
            )

        nrow = self.hfield_nrow
        ncol = self.hfield_ncol
        x = np.linspace(-self.hfield_size_x, self.hfield_size_x, ncol)
        y = np.linspace(-self.hfield_size_y, self.hfield_size_y, nrow)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        blend = np.clip((R - flat_radius) / 0.25, 0.0, 1.0)
        elevation = np.zeros((nrow, ncol), dtype=np.float32)

        if mode == "rough":
            freq1 = np.random.uniform(2.5, 3.5)
            freq2 = np.random.uniform(4.5, 6.0)
            phase_x = np.random.uniform(0, 2 * math.pi)
            phase_y = np.random.uniform(0, 2 * math.pi)
            wave = 0.5 * (np.sin(freq1 * X + phase_x) * np.cos(freq1 * Y + phase_y) + 1.0)
            fine = 0.5 * (np.sin(freq2 * X) * np.sin(freq2 * Y) + 1.0)
            elevation = (0.7 * wave + 0.3 * fine) * difficulty * 0.50
        elif mode == "wavy":
            wavelength = np.random.uniform(1.2, 1.8)
            freq = 2 * math.pi / wavelength
            angle = np.random.uniform(0, 2 * math.pi)
            proj = X * math.cos(angle) + Y * math.sin(angle)
            elevation = 0.5 * (np.sin(freq * proj) + 1.0) * difficulty * 0.55
        elif mode == "slopes":
            angle = np.random.uniform(0, 2 * math.pi)
            grade = np.random.uniform(0.02, 0.04) * difficulty
            ramp = np.clip(((X - center_x) * math.cos(angle) + (Y - center_y) * math.sin(angle)) * grade, -0.02, 0.02)
            elevation = (ramp + 0.02) / self.hfield_max_z
        elif mode == "mixed":
            w_wave = 0.5 * (np.sin(3.0 * X) * np.cos(2.5 * Y) + 1.0)
            w_noise = 0.5 * (np.sin(6.0 * X) * np.sin(5.5 * Y) + 1.0)
            elevation = (0.6 * w_wave + 0.4 * w_noise) * difficulty * 0.50
        else:  # flat
            elevation = np.zeros((nrow, ncol), dtype=np.float32)

        final_elevation = np.clip(blend * elevation, 0.0, 1.0)
        self.model.hfield_data[:] = final_elevation.flatten()

        mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"material_{mode}")
        if mat_id != -1 and self.floor_geom_id != -1:
            self.model.geom_matid[self.floor_geom_id] = mat_id

    # ================================================================
    # CPG (Central Pattern Generator)
    # ================================================================
    def compute_cpg(self, phi: float, vx: float, vy: float, wz: float) -> np.ndarray:
        """Unified natural stance CPG with foot clearance, smooth weight transfer, and balanced rotation."""
        speed = math.sqrt(vx * vx + vy * vy) + abs(wz) * 0.1
        base_hr = 0.070  # Hip roll for opened-outward stance (15.3cm width)
        base_hy = 0.035  # Toe-out yaw

        if speed < 0.025:
            # Natural upright standing pose: straight vertical torso, flat feet, opened outward
            return np.array([
                -base_hy,  base_hr, -0.09, 0.32, -0.23, -base_hr,
                 base_hy, -base_hr, -0.09, 0.32, -0.23,  base_hr
            ], dtype=np.float32)

        sin_p = math.sin(phi)
        swing_l = max(0.0, sin_p)
        swing_r = max(0.0, -sin_p)

        # Pitch & Knee Kinematics (Deeper Bent-Leg Crouch Locomotion)
        is_pure_strafe = (abs(vx) < 0.03 and abs(vy) > 0.02)
        is_pure_rotate = (abs(vx) < 0.02 and abs(vy) < 0.02 and abs(wz) > 0.05)

        if is_pure_strafe:
            # Zero-drift lateral strafing: cancels sagittal backward reaction force across all speeds
            nom_pitch = -0.045
            knee_nom = 0.30
            step_p = 0.028 if vy >= 0.0 else 0.030
            clearance_pitch = 0.0
            knee_a = 0.30
            stance_straighten = 0.0
        elif is_pure_rotate:
            # Pure in-place rotation on the spot (Zero linear translation)
            nom_pitch = -0.020
            knee_nom = 0.28
            knee_a = 0.28
            step_p = 0.0
            clearance_pitch = 0.0
            stance_straighten = 0.0
        elif vx >= 0.0:
            # Smooth transition from standing (-0.09) to natural bent-knee forward walking
            fwd_blend = math.tanh(3.0 * vx)
            nom_pitch = -0.07 - 0.06 * fwd_blend
            knee_nom = 0.30 + 0.06 * fwd_blend
            # Positive minimum stride threshold ensures robot advances forward at all speeds
            step_p = np.clip(0.06 + 0.65 * vx, 0.06, 0.22)
            clearance_pitch = -0.015 * fwd_blend
            knee_a = np.clip(0.38 + 0.10 * abs(vx) + 0.15 * abs(vy), 0.34, 0.48)
            stance_straighten = 0.025 * fwd_blend
        else:
            # Natural upright backward walking
            nom_pitch = -0.09 + 0.03 * vx
            knee_nom = 0.30
            step_p = np.clip(0.30 * vx, -0.12, 0.0)
            clearance_pitch = 0.02 * abs(vx)
            knee_a = np.clip(0.30 + 0.06 * abs(vx), 0.25, 0.35)
            stance_straighten = 0.0

        l_toe_lift = clearance_pitch * swing_l
        r_toe_lift = clearance_pitch * swing_r

        lk = (knee_nom - stance_straighten * swing_r) + knee_a * swing_l
        rk = (knee_nom - stance_straighten * swing_l) + knee_a * swing_r

        if is_pure_rotate:
            # Completely symmetric pitch: zero forward/backward translation
            lhp = nom_pitch
            rhp = nom_pitch
        else:
            lhp = nom_pitch - step_p * sin_p
            rhp = nom_pitch + step_p * sin_p

        lap = -(lhp + lk) + l_toe_lift
        rap = -(rhp + rk) + r_toe_lift

        # Lateral weight transfer and sway
        if is_pure_rotate:
            # Symmetric lateral sway: zero sideways drift
            sway = -0.012 * sin_p
            lhr = np.clip(base_hr + sway, -0.08, 0.25)
            rhr = np.clip(-base_hr + sway, -0.25, 0.08)
        elif is_pure_strafe:
            sway = -(0.035 + 0.06 * abs(vy)) * sin_p
            if vy >= 0.0:
                lat_left = np.clip(1.20 * vy, 0.0, 0.18) * swing_l
                lat_right = np.clip(0.40 * vy, 0.0, 0.05) * swing_r
            else:
                lat_left = np.clip(0.40 * vy, -0.05, 0.0) * swing_l
                lat_right = np.clip(1.20 * vy, -0.18, 0.0) * swing_r
            lhr = np.clip(base_hr + sway + lat_left, -0.08, 0.25)
            rhr = np.clip(-base_hr + sway + lat_right, -0.25, 0.08)
        else:
            sway = -(0.032 + 0.010 * abs(vx) + 0.015 * abs(vy)) * sin_p
            if vy >= 0.0:
                lat_left = np.clip(1.20 * vy, 0.0, 0.18) * swing_l
                lat_right = np.clip(0.40 * vy, 0.0, 0.05) * swing_r
            else:
                lat_left = np.clip(0.40 * vy, -0.05, 0.0) * swing_l
                lat_right = np.clip(1.20 * vy, -0.18, 0.0) * swing_r
            lhr = np.clip(base_hr + sway + lat_left, -0.08, 0.25)
            rhr = np.clip(-base_hr + sway + lat_right, -0.25, 0.08)

        lar = -lhr
        rar = -rhr

        # Yaw stepping for rotation
        if is_pure_rotate:
            yaw_step = np.clip(0.45 * wz, -0.25, 0.25)
        else:
            yaw_step = np.clip(0.32 * wz, -0.22, 0.22)
        lhy = -base_hy + yaw_step * swing_l
        rhy = base_hy + yaw_step * swing_r

        return np.array([lhy, lhr, lhp, lk, lap, lar, rhy, rhr, rhp, rk, rap, rar], dtype=np.float32)

    # ================================================================
    # COMMAND SAMPLING
    # ================================================================
    def set_command(self, vx: float, vy: float, wz: float):
        self.command = np.array([vx, vy, wz], dtype=np.float32)

    def _sample_command(self):
        """Domain randomization matching user goal: pure straight motions, in-place rotation, and walk/trot."""
        mode = np.random.choice([
            "forward", "backward", "strafe_left", "strafe_right",
            "turn_in_place_ccw", "turn_in_place_cw", "combined", "stop"
        ], p=[0.24, 0.16, 0.12, 0.12, 0.10, 0.10, 0.06, 0.10])

        if mode == "forward":
            # Mostly stable walk (0.08 - 0.25 m/s), 15% fast walk/trot (0.25 - 0.35 m/s)
            vx = np.random.uniform(0.25, 0.35) if np.random.rand() < 0.15 else np.random.uniform(0.08, 0.25)
            self.command = np.array([vx, 0.0, 0.0], dtype=np.float32)
        elif mode == "backward":
            self.command = np.array([np.random.uniform(-0.16, -0.06), 0.0, 0.0], dtype=np.float32)
        elif mode == "strafe_left":
            self.command = np.array([0.0, np.random.uniform(0.06, 0.16), 0.0], dtype=np.float32)
        elif mode == "strafe_right":
            self.command = np.array([0.0, np.random.uniform(-0.16, -0.06), 0.0], dtype=np.float32)
        elif mode == "turn_in_place_ccw":
            self.command = np.array([0.0, 0.0, np.random.uniform(0.18, 0.45)], dtype=np.float32)
        elif mode == "turn_in_place_cw":
            self.command = np.array([0.0, 0.0, np.random.uniform(-0.45, -0.18)], dtype=np.float32)
        elif mode == "combined":
            fwd_or_bwd = np.random.choice([1.0, -1.0], p=[0.6, 0.4])
            vx = np.random.uniform(0.08, 0.22) if fwd_or_bwd > 0 else np.random.uniform(-0.14, -0.06)
            self.command = np.array([
                vx,
                np.random.uniform(-0.10, 0.10),
                np.random.uniform(-0.25, 0.25)
            ], dtype=np.float32)
        else:
            self.command = np.zeros(3, dtype=np.float32)

        self.command_duration = np.random.randint(120, 260)

    # ================================================================
    # RESET
    # ================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Curriculum: auto-scale difficulty based on survival rate
        if self.training and self.enable_curriculum:
            self.episode_count += 1
            if len(self.recent_survivals) >= 50:
                survival_rate = np.mean(self.recent_survivals[-50:])
                if survival_rate > 0.80:
                    self.curriculum_difficulty = min(0.8, self.curriculum_difficulty + 0.01)
                elif survival_rate < 0.40:
                    self.curriculum_difficulty = max(0.15, self.curriculum_difficulty - 0.01)
                self.terrain_difficulty = self.curriculum_difficulty

        # Generate terrain
        if self.training:
            self.generate_terrain(mode=self.terrain_type, difficulty=self.terrain_difficulty)

        self.phase = np.random.uniform(0.0, 2.0 * math.pi) if self.training else 0.0
        self.step_counter = 0
        self.push_steps = 0
        self.push_active = False
        self.prev_tilt_magnitude = 0.0
        self.target_yaw = 0.0
        self.filtered_p_rate = 0.0
        self.filtered_r_rate = 0.0
        self.prev_ctrl_targets = None

        if self.training:
            self._sample_command()
            self.target_v = self.command.copy()
        else:
            self.target_v = np.zeros(3, dtype=np.float32)

        init_cpg = self.compute_cpg(self.phase, *self.target_v)
        for i, qpos_adr in enumerate(self.leg_qpos_indices):
            noise = np.random.uniform(-0.02, 0.02) if self.training else 0.0
            self.data.qpos[qpos_adr] = init_cpg[i] + noise

        for act_id, (qpos_adr, val) in self.upper_act_dict.items():
            self.data.qpos[qpos_adr] = val
            self.data.ctrl[act_id] = val

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.initial_root_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0

        # ============================================================
        # DOMAIN RANDOMIZATION: Initial orientation perturbation
        # Train the robot to recover from slightly tilted starting poses
        # ============================================================
        if self.training and np.random.rand() < 0.3:
            tilt_pitch = np.random.uniform(-0.06, 0.06)  # ~3.4 deg
            tilt_roll = np.random.uniform(-0.04, 0.04)   # ~2.3 deg
            # Apply small quaternion rotation
            qx = math.sin(tilt_roll / 2)
            qy = math.sin(tilt_pitch / 2)
            self.data.qpos[4] = qx
            self.data.qpos[5] = qy
            w_sq = max(0.0, 1.0 - qx * qx - qy * qy)
            self.data.qpos[3] = math.sqrt(w_sq)

        self.data.qvel[:] = 0.0

        # Apply initial velocity perturbation for balance training
        if self.training and np.random.rand() < 0.2:
            self.data.qvel[0] = np.random.uniform(-0.08, 0.08)  # forward/back
            self.data.qvel[1] = np.random.uniform(-0.06, 0.06)  # lateral

        for i, act_id in enumerate(self.leg_act_indices):
            self.data.ctrl[act_id] = self.data.qpos[self.leg_qpos_indices[i]]

        mujoco.mj_forward(self.model, self.data)
        self.prev_action = np.zeros(self.num_leg_joints, dtype=np.float32)

        return self._get_obs(), {}

    # ================================================================
    # OBSERVATION
    # ================================================================
    def _get_obs(self):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ], dtype=np.float32)

        projected_gravity = -R[2, :]
        local_angvel = R.T @ self.data.qvel[3:6]
        local_linvel = R.T @ self.data.qvel[0:3]

        cpg_target = self.compute_cpg(self.phase, *self.target_v)
        curr_joint_pos = np.array([self.data.qpos[idx] for idx in self.leg_qpos_indices], dtype=np.float32)
        rel_joint_pos = curr_joint_pos - cpg_target
        joint_vel = np.array([self.data.qvel[idx] for idx in self.leg_qvel_indices], dtype=np.float32)

        sin_phase = math.sin(self.phase)
        cos_phase = math.cos(self.phase)
        r_contact, l_contact, _ = self._check_contacts()

        current_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        if abs(self.target_v[2]) < 0.02:
            yaw_diff = (current_yaw - self.target_yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            yaw_diff = 0.0
            self.target_yaw = current_yaw

        heading_feat = np.array([math.sin(yaw_diff), math.cos(yaw_diff)], dtype=np.float32)

        # ============================================================
        # NEW: IMU tilt error signal
        # projected_gravity[0] = forward tilt, [1] = lateral tilt
        # Ideal upright: gravity = [0, 0, -1], so tilt_error ≈ [gx, gy]
        # ============================================================
        tilt_error = np.array([projected_gravity[0], projected_gravity[1]], dtype=np.float32)

        # Tilt angular velocity (pitch rate, roll rate) in local frame
        tilt_rate = np.array([local_angvel[1], local_angvel[0]], dtype=np.float32)

        # Terrain slope under torso
        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        slope_x, slope_y = self.get_terrain_slope(torso_x, torso_y)
        terrain_slope = np.array([slope_x, slope_y], dtype=np.float32)

        # Push indicator: tells policy whether it's being pushed right now
        push_indicator = np.array([
            float(self.push_active),
            float(self.push_steps > 0) * np.sign(np.sum(self.push_force[:2]))
        ], dtype=np.float32)

        obs = np.concatenate([
            self.target_v,                                               # 3
            projected_gravity,                                           # 3
            local_angvel * 0.25,                                         # 3
            local_linvel * 2.0,                                          # 3
            rel_joint_pos,                                               # 12
            joint_vel * 0.05,                                            # 12
            self.prev_action,                                            # 12
            np.array([sin_phase, cos_phase], dtype=np.float32),          # 2
            np.array([float(r_contact), float(l_contact)], dtype=np.float32),  # 2
            heading_feat,                                                # 2
            tilt_error,                                                  # 2 (NEW)
            tilt_rate * 0.25,                                            # 2 (NEW)
            terrain_slope * 5.0,                                         # 2 (NEW)
            push_indicator,                                              # 2 (NEW)
        ]).astype(np.float32)

        return obs

    # ================================================================
    # CONTACTS
    # ================================================================
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

    # ================================================================
    # STEP
    # ================================================================
    def step(self, action):
        self.step_counter += 1
        action = np.clip(action, -1.0, 1.0)

        if self.training:
            self.command_duration -= 1
            if self.command_duration <= 0:
                self._sample_command()

            # ============================================================
            # AGGRESSIVE PUSH DISTURBANCE SCHEDULE
            # Phase 1 (steps 0-200): Gentle pushes to learn basic balance
            # Phase 2 (steps 200+): Stronger pushes, more frequent
            # Also apply random lateral impulses for IMU correction training
            # ============================================================
            push_prob = 0.015 if self.step_counter < 200 else 0.025
            max_force = 3.5 if self.step_counter < 200 else 6.0

            if np.random.rand() < push_prob and self.push_steps == 0:
                self.push_steps = np.random.randint(3, 15)
                # Random direction push with bias toward forward/backward
                angle = np.random.uniform(0, 2 * math.pi)
                magnitude = np.random.uniform(1.5, max_force)
                self.push_force[0] = magnitude * math.cos(angle)
                self.push_force[1] = magnitude * math.sin(angle)
                self.push_active = True

        # Smooth command interpolation
        max_dv = 0.5 * self.dt
        dv = np.clip(self.command - self.target_v, -max_dv, max_dv)
        self.target_v += dv

        # ============================================================
        # UNITREE-STYLE RESPONSIVE TILT CORRECTION & FALL ARREST
        # ============================================================
        q = self.data.qpos[3:7]
        w, x, y, z = q
        curr_pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
        curr_roll = math.atan2(float(2.0 * (w * x + y * z)), float(1.0 - 2.0 * (x * x + y * y)))
        curr_yaw = math.atan2(float(2.0 * (w * z + x * y)), float(1.0 - 2.0 * (y * y + z * z)))

        nom_pitch = -0.025  # ~ -1.4 deg natural upright pitch
        target_pitch = nom_pitch + (0.06 * math.tanh(2.5 * self.target_v[0]) if self.target_v[0] > 0 else -0.03 * math.tanh(2.5 * abs(self.target_v[0])))
        p_err = curr_pitch - target_pitch
        raw_p_rate = float(self.data.qvel[4])
        raw_r_rate = float(self.data.qvel[3])
        self.filtered_p_rate = 0.50 * self.filtered_p_rate + 0.50 * raw_p_rate
        self.filtered_r_rate = 0.50 * self.filtered_r_rate + 0.50 * raw_r_rate
        p_rate = self.filtered_p_rate
        r_err = curr_roll
        r_rate = self.filtered_r_rate

        # Continuous phase clock advances without freezing during walking commands
        cmd_speed = math.sqrt(self.target_v[0] ** 2 + self.target_v[1] ** 2) + abs(self.target_v[2]) * 0.1
        if cmd_speed > 0.02:
            # Dynamic frequency scaling: smoothly scales gait frequency up to 1.9 Hz at higher forward speeds
            eff_freq = self.gait_frequency + 2.0 * max(0.0, self.target_v[0] - 0.14)
            self.phase = (self.phase + 2.0 * math.pi * eff_freq * self.dt) % (2.0 * math.pi)

        cpg_targets = self.compute_cpg(self.phase, self.target_v[0], self.target_v[1], self.target_v[2])

        # Velocity gating with dynamic balance override (keeps PPO policy active if tilted)
        tilt_mag = math.sqrt(p_err ** 2 + r_err ** 2)
        tilt_active = min(1.0, max(0.0, (tilt_mag - 0.03) / 0.04))
        action_weight = max(min(1.0, max(0.0, (cmd_speed - 0.012) / 0.020)), tilt_active)
        final_targets = cpg_targets + (self.action_scale * action_weight) * action

        # Progressive non-linear scaling for rapid fall arrest (with deadband to eliminate chatter during normal gait)
        p_boost = 1.4 * p_err * abs(p_err) if abs(p_err) > 0.02 else 0.0
        r_boost = 1.4 * r_err * abs(r_err) if abs(r_err) > 0.02 else 0.0

        is_pure_rotate = (abs(self.target_v[0]) < 0.02 and abs(self.target_v[1]) < 0.02 and abs(self.target_v[2]) > 0.05)
        if is_pure_rotate:
            corr_pa = float(np.clip(0.25 * p_err + 0.025 * p_rate, -0.12, 0.12))
            dcm_p = 0.0
            lat_step_adj = 0.0
        else:
            corr_pa = float(np.clip(0.40 * (p_err + p_boost) + 0.035 * p_rate, -0.26, 0.26))
            # Dynamic step displacement to place swing foot under CoM
            dcm_p = float(np.clip(-(0.50 * (p_err + p_boost) + 0.055 * p_rate), -0.20, 0.20))
            # Lateral Capture Point (Preserved from calibrated lateral balance)
            lat_step_adj = float(np.clip(0.20 * r_err + 0.025 * r_rate, -0.06, 0.06))

        # 2. Stance Leg Ground Reaction Torques (Ankle Pitch & Roll, Hip Roll)
        corr_ra = float(np.clip(0.35 * r_err + 0.030 * r_rate, -0.12, 0.12))
        corr_rh = float(np.clip(-0.25 * r_err - 0.030 * r_rate, -0.12, 0.12))

        final_targets[4] += corr_pa   # left_ankle_pitch
        final_targets[10] += corr_pa  # right_ankle_pitch

        final_targets[1] += corr_rh   # left_hip_roll
        final_targets[5] += corr_ra   # left_ankle_roll
        final_targets[7] += corr_rh   # right_hip_roll
        final_targets[11] += corr_ra  # right_ankle_roll

        # 3. SWING LEG DYNAMIC CAPTURE POINT (DCM Stepping Strategy - Unitree Fall Prevention)
        sin_p = math.sin(self.phase)
        swing_l = max(0.0, sin_p)
        swing_r = max(0.0, -sin_p)

        final_targets[2] += dcm_p * swing_l
        final_targets[4] -= dcm_p * swing_l
        final_targets[8] += dcm_p * swing_r
        final_targets[10] -= dcm_p * swing_r

        final_targets[1] += lat_step_adj * swing_l
        final_targets[7] += lat_step_adj * swing_r

        # 4. ACTIVE CLOSED-LOOP HEADING LOCK (Zero Drift Yaw Steering)
        if abs(self.target_v[2]) < 0.02:
            yaw_err = (curr_yaw - self.target_yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw_rate = float(self.data.qvel[5])
            corr_yaw = float(np.clip(-(0.40 * yaw_err + 0.06 * yaw_rate), -0.16, 0.16))
            final_targets[0] += corr_yaw * swing_l
            final_targets[6] += corr_yaw * swing_r
        else:
            self.target_yaw = curr_yaw

        if self.prev_ctrl_targets is None:
            self.prev_ctrl_targets = final_targets.copy()
        smoothed_targets = 0.80 * final_targets + 0.20 * self.prev_ctrl_targets
        self.prev_ctrl_targets = smoothed_targets.copy()

        for i, act_id in enumerate(self.leg_act_indices):
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(smoothed_targets[i], ctrl_min, ctrl_max)

        # 5. Dynamic Arm Swing with Unitree Inertial Restoring Counter-Torque
        arm_amp = 0.22 * math.tanh(3.0 * self.target_v[0])
        arm_p_bal = float(np.clip(-0.50 * (p_err + p_boost) - 0.040 * p_rate, -0.35, 0.35))
        for act_id, (_, target) in self.upper_act_dict.items():
            if act_id == self.r_shoulder_pitch_id:
                self.data.ctrl[act_id] = target - arm_amp * sin_p + arm_p_bal
            elif act_id == self.l_shoulder_pitch_id:
                self.data.ctrl[act_id] = target + arm_amp * sin_p + arm_p_bal
            else:
                self.data.ctrl[act_id] = target

        # Apply push forces
        if self.push_steps > 0:
            self.data.xfrc_applied[self.torso_body_id, :2] = self.push_force[:2]
            self.push_steps -= 1
            if self.push_steps == 0:
                self.push_active = False
        else:
            self.data.xfrc_applied[self.torso_body_id, :2] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = bool(self.step_counter >= self.max_episode_steps)

        # Track survival for curriculum
        if self.training and (terminated or truncated):
            survived = not terminated
            self.recent_survivals.append(1.0 if survived else 0.0)
            if len(self.recent_survivals) > 200:
                self.recent_survivals = self.recent_survivals[-200:]

        self.prev_action = action.copy()
        obs = self._get_obs()

        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        terrain_z = self.get_terrain_height(torso_x, torso_y)

        info = {
            "cmd_vx": self.target_v[0],
            "cmd_vy": self.target_v[1],
            "cmd_wz": self.target_v[2],
            "act_vx": self.data.qvel[0],
            "act_vy": self.data.qvel[1],
            "act_wz": self.data.qvel[5],
            "torso_z": torso_z,
            "terrain_z": terrain_z,
            "rel_torso_z": torso_z - terrain_z,
            "push_active": self.push_active,
            "curriculum_difficulty": self.curriculum_difficulty if self.training else self.terrain_difficulty,
        }

        return obs, reward, terminated, truncated, info

    # ================================================================
    # REWARD FUNCTION
    # ================================================================
    def _compute_reward(self, action):
        q = self.data.qpos[3:7]
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ], dtype=np.float32)
        local_linvel = R.T @ self.data.qvel[0:3]
        local_angvel = R.T @ self.data.qvel[3:6]
        projected_gravity = -R[2, :]

        # ============================================================
        # 1. VELOCITY TRACKING
        # ============================================================
        err_vx = local_linvel[0] - self.target_v[0]
        r_vx = 12.0 * math.exp(-err_vx ** 2 / 0.012)

        err_vy = local_linvel[1] - self.target_v[1]
        r_vy = 12.0 * math.exp(-err_vy ** 2 / 0.012)

        err_wz = local_angvel[2] - self.target_v[2]
        r_wz = 12.0 * math.exp(-err_wz ** 2 / 0.018)

        # Forward/Backward propulsion bonus
        r_fwd = 0.0
        p_stuck_x = 0.0
        if self.target_v[0] > 0.03:
            if local_linvel[0] > 0.03:
                r_fwd = 35.0 * min(local_linvel[0], self.target_v[0])
            else:
                p_stuck_x = 30.0 * (self.target_v[0] - max(0.0, local_linvel[0]))
        elif self.target_v[0] < -0.03:
            if local_linvel[0] < -0.03:
                r_fwd = 35.0 * min(-local_linvel[0], -self.target_v[0])
            else:
                p_stuck_x = 30.0 * (-self.target_v[0] - max(-0.02, -local_linvel[0]))

        # Lateral/rotation propulsion
        r_lat = 0.0
        if abs(self.target_v[1]) > 0.03:
            match = local_linvel[1] * np.sign(self.target_v[1])
            if match > 0.02:
                r_lat = 25.0 * min(abs(local_linvel[1]), abs(self.target_v[1]))

        r_rot = 0.0
        if abs(self.target_v[2]) > 0.08:
            if abs(local_angvel[2]) > 0.04 and (local_angvel[2] * self.target_v[2] > 0):
                r_rot = 20.0 * min(abs(local_angvel[2]), abs(self.target_v[2]))

        # ============================================================
        # 2. ZERO-DRIFT PENALTIES (when commanding stop)
        # ============================================================
        p_lat_drift = 30.0 * (local_linvel[1] ** 2) if abs(self.target_v[1]) < 0.02 else 0.0
        p_fwd_drift = 30.0 * (local_linvel[0] ** 2) if abs(self.target_v[0]) < 0.02 else 0.0
        p_yaw_drift = 25.0 * (local_angvel[2] ** 2) if abs(self.target_v[2]) < 0.02 else 0.0

        # ============================================================
        # 3. HEADING LOCK
        # ============================================================
        current_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        r_heading = 0.0
        p_heading = 0.0
        if abs(self.target_v[2]) < 0.02:
            yaw_diff = (current_yaw - self.target_yaw + math.pi) % (2 * math.pi) - math.pi
            r_heading = 6.0 * math.exp(-yaw_diff ** 2 / 0.02)
            p_heading = 20.0 * (yaw_diff ** 2)
        else:
            self.target_yaw = current_yaw

        p_turn_drift = 0.0
        if abs(self.target_v[2]) > 0.08 and abs(self.target_v[0]) < 0.03 and abs(self.target_v[1]) < 0.03:
            p_turn_drift = 25.0 * (local_linvel[0] ** 2 + local_linvel[1] ** 2)

        # ============================================================
        # 4. UPRIGHT POSTURE & HEIGHT
        # ============================================================
        tilt_magnitude = math.sqrt(projected_gravity[0] ** 2 + projected_gravity[1] ** 2)
        r_upright = 10.0 * math.exp(-(tilt_magnitude ** 2) / 0.01)
        p_tilt = 20.0 * (tilt_magnitude ** 2)

        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        terrain_z = self.get_terrain_height(torso_x, torso_y)
        rel_torso_z = torso_z - terrain_z

        r_height = 8.0 * math.exp(-((rel_torso_z - self.nominal_torso_height) ** 2) / 0.005)
        p_sag = 25.0 * (max(0.0, 0.35 - rel_torso_z) ** 2)

        # ============================================================
        # 5. IMU TILT CORRECTION REWARD (NEW)
        # Actively rewards the robot for REDUCING tilt over time.
        # If tilt was large on previous step and is now smaller,
        # give a large bonus. This trains push recovery behavior.
        # ============================================================
        r_tilt_recovery = 0.0
        tilt_improvement = self.prev_tilt_magnitude - tilt_magnitude
        if tilt_improvement > 0.005:
            # Robot is actively recovering from a tilt - big reward!
            r_tilt_recovery = 25.0 * tilt_improvement
        elif tilt_magnitude > 0.15:
            # Robot is tilted but not recovering - penalty
            r_tilt_recovery = -8.0 * tilt_magnitude

        # ============================================================
        # 6. PUSH RECOVERY BONUS (NEW)
        # Extra reward for maintaining upright posture during/after push
        # ============================================================
        r_push_recovery = 0.0
        if self.push_active or self.push_steps > 0:
            # During push: reward staying upright
            if tilt_magnitude < 0.15:
                r_push_recovery = 8.0
            elif tilt_magnitude < 0.25:
                r_push_recovery = 4.0
        elif self.prev_tilt_magnitude > 0.1 and tilt_magnitude < 0.05:
            # Just recovered from perturbation: big bonus
            r_push_recovery = 12.0

        self.prev_tilt_magnitude = tilt_magnitude

        # ============================================================
        # 7. FOOT CONTACT & FLATNESS
        # ============================================================
        r_contact_flag, l_contact_flag, non_foot = self._check_contacts()

        # Foot flatness
        rf_mat = self.data.xmat[self.r_foot_bid].reshape(3, 3)
        lf_mat = self.data.xmat[self.l_foot_bid].reshape(3, 3)
        rf_tilt2 = rf_mat[0, 2] ** 2 + rf_mat[1, 2] ** 2
        lf_tilt2 = lf_mat[0, 2] ** 2 + lf_mat[1, 2] ** 2
        stance_tilt = rf_tilt2 * float(r_contact_flag) + lf_tilt2 * float(l_contact_flag)
        r_flat_foot = 8.0 * math.exp(-stance_tilt / 0.005)
        p_foot_tilt = 25.0 * stance_tilt

        # Anti-wobble
        p_roll_wobble = 2.0 * (local_angvel[0] ** 2)

        # Foot clearance during swing
        r_clearance = 0.0
        cmd_speed = math.sqrt(self.target_v[0] ** 2 + self.target_v[1] ** 2)
        if cmd_speed > 0.03:
            sin_phase = math.sin(self.phase)
            lf_z = self.data.site_xpos[self.left_foot_site_id, 2]
            rf_z = self.data.site_xpos[self.right_foot_site_id, 2]
            lf_rel_z = lf_z - self.get_terrain_height(
                self.data.site_xpos[self.left_foot_site_id, 0],
                self.data.site_xpos[self.left_foot_site_id, 1]
            )
            rf_rel_z = rf_z - self.get_terrain_height(
                self.data.site_xpos[self.right_foot_site_id, 0],
                self.data.site_xpos[self.right_foot_site_id, 1]
            )
            if sin_phase > 0.35:
                r_clearance += 3.0 * min(max(0.0, (lf_rel_z - 0.008) / 0.040), 1.0)
            elif sin_phase < -0.35:
                r_clearance += 3.0 * min(max(0.0, (rf_rel_z - 0.008) / 0.040), 1.0)

        # Foot slip penalty
        lf_vel = np.zeros(6)
        rf_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.left_foot_site_id, lf_vel, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.right_foot_site_id, rf_vel, 0)
        lf_vxy2 = lf_vel[3] ** 2 + lf_vel[4] ** 2
        rf_vxy2 = rf_vel[3] ** 2 + rf_vel[4] ** 2
        p_slip = 0.3 * (lf_vxy2 * float(l_contact_flag) + rf_vxy2 * float(r_contact_flag))

        # Non-foot contact penalty (knees/body touching ground)
        p_non_foot = 10.0 if non_foot else 0.0

        # ============================================================
        # 8. SMOOTHNESS & ENERGY EFFICIENCY
        # ============================================================
        p_smoothness = 0.03 * np.sum(np.square(action - self.prev_action))
        p_torque = 1e-4 * np.sum(np.square(self.data.qfrc_actuator[self.leg_qvel_indices]))
        p_gyro = 0.15 * np.sum(np.square(local_angvel))

        # ============================================================
        # 9. STANCE SYMMETRY (natural gait)
        # Reward symmetric gait timing when walking straight
        # ============================================================
        r_symmetry = 0.0
        if cmd_speed > 0.04 and abs(self.target_v[2]) < 0.05:
            # Both feet should have similar contact duration
            sin_phase = math.sin(self.phase)
            # Reward alternating contacts (natural gait pattern)
            if abs(sin_phase) > 0.5:
                expected_l = sin_phase > 0  # left swing when sin > 0
                if expected_l and not l_contact_flag and r_contact_flag:
                    r_symmetry = 2.0
                elif not expected_l and not r_contact_flag and l_contact_flag:
                    r_symmetry = 2.0

        # ============================================================
        # 10. SURVIVAL BONUS
        # ============================================================
        r_healthy = 3.5

        # ============================================================
        # TOTAL REWARD
        # ============================================================
        total_reward = (
            r_vx + r_vy + r_wz +
            r_fwd + r_lat + r_rot +
            r_heading +
            r_upright + r_height +
            r_tilt_recovery +
            r_push_recovery +
            r_clearance +
            r_flat_foot +
            r_symmetry +
            r_healthy -
            p_stuck_x -
            p_lat_drift - p_fwd_drift - p_yaw_drift -
            p_heading -
            p_turn_drift -
            p_tilt -
            p_foot_tilt -
            p_roll_wobble -
            p_gyro -
            p_smoothness -
            p_torque -
            p_slip -
            p_sag -
            p_non_foot
        )

        return total_reward

    # ================================================================
    # TERMINATION
    # ================================================================
    def _is_terminated(self):
        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        terrain_z = self.get_terrain_height(torso_x, torso_y)
        rel_torso_z = torso_z - terrain_z

        # Fall detection relative to local terrain height
        if rel_torso_z < 0.20 or rel_torso_z > 0.55:
            return True

        # Fall detection by tilt (> 60 deg from vertical)
        q = self.data.qpos[3:7]
        w, x, y, z = q
        gz = -(1.0 - 2.0 * (x ** 2 + y ** 2))
        if gz > -0.50:
            return True

        return False
