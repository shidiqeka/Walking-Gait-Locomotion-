import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class BarelangSTerrainEnv(gym.Env):
    """
    MuJoCo 3D Procedural Terrain Locomotion Environment for Barelang S Humanoid.
    Supports Rough/Bumpy Terrain, Rolling Waves, Gentle Slopes, Stepping Stones, and Mixed Terrain.
    Fully compatible with 54-dim observation space for seamless transfer learning from flat policy.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, 
                 model_path="barelangS_terrain_mujoco.xml", 
                 training=True, 
                 terrain_type="random", 
                 terrain_difficulty=0.6):
        super().__init__()
        self.training = training
        self.terrain_type = terrain_type
        self.terrain_difficulty = terrain_difficulty
        self.dt = 0.02 # 50 Hz control loop
        self.frame_skip = 10 # 500 Hz internal physics (0.002s * 10)
        self.max_episode_steps = 1000 # 20.0 seconds max per episode

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
        # size: [radius_x, radius_y, elevation_z, base_z]
        self.hfield_size_x = self.model.hfield_size[0, 0] # 6.0m
        self.hfield_size_y = self.model.hfield_size[0, 1] # 6.0m
        self.hfield_max_z  = self.model.hfield_size[0, 2] # 0.05m (5 cm)

        # Joint name definitions
        self.leg_joint_names = [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll"
        ]
        self.num_leg_joints = len(self.leg_joint_names)
        self.leg_qpos_indices = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.leg_joint_names]
        self.leg_qvel_indices = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.leg_joint_names]
        self.leg_act_indices  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.leg_joint_names]

        # Passive upper body posture
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

        # Residual action space: normalized offsets in [-1, 1] scaled by 0.14 rad
        self.action_scale = 0.14
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_leg_joints,), dtype=np.float32
        )

        # 54-dimensional Observation space
        obs_dim = 3 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 2 + 2 # 54
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Key site & body identifiers
        self.torso_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torso")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        self.right_foot_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.left_foot_site_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.r_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll")
        self.l_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Kinematic & CPG constants
        self.gait_frequency = 1.6 # Hz
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

        # Push disturbance variables
        self.push_steps = 0
        self.push_force = np.zeros(3, dtype=np.float32)

        # Generate initial terrain
        self.generate_terrain(mode=self.terrain_type, difficulty=self.terrain_difficulty)

    def get_terrain_height(self, x: float, y: float) -> float:
        """Computes continuous bilinear interpolated terrain elevation at world (x, y)."""
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

        z_norm = (1 - du) * (1 - dv) * z00 + du * (1 - du) * z10 + (1 - du) * dv * z01 + du * dv * z11
        return float(z_norm * self.hfield_max_z)

    def generate_terrain(self, mode="random", difficulty=0.6, flat_radius=0.25, center_x=0.0, center_y=0.0):
        """Generates procedural 3D terrain into MuJoCo hfield_data with flat spawn/transition area."""
        if mode == "random":
            mode = np.random.choice(["rough", "wavy", "slopes", "mixed", "flat"], p=[0.35, 0.25, 0.20, 0.15, 0.05])

        nrow = self.hfield_nrow
        ncol = self.hfield_ncol
        x = np.linspace(-self.hfield_size_x, self.hfield_size_x, ncol)
        y = np.linspace(-self.hfield_size_y, self.hfield_size_y, nrow)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        # Smooth flat circle at center so robot remains stable during spawn or terrain transition
        blend = np.clip((R - flat_radius) / 0.25, 0.0, 1.0)

        elevation = np.zeros((nrow, ncol), dtype=np.float32)

        if mode == "rough":
            # Multi-scale sinusoidal & random gravel bumps
            freq1 = np.random.uniform(2.5, 3.5)
            freq2 = np.random.uniform(4.5, 6.0)
            phase_x = np.random.uniform(0, 2*math.pi)
            phase_y = np.random.uniform(0, 2*math.pi)
            
            wave = 0.5 * (np.sin(freq1 * X + phase_x) * np.cos(freq1 * Y + phase_y) + 1.0)
            fine = 0.5 * (np.sin(freq2 * X) * np.sin(freq2 * Y) + 1.0)
            elevation = (0.7 * wave + 0.3 * fine) * difficulty * 0.50 # max ~2.5 cm

        elif mode == "wavy":
            # Rolling hills
            wavelength = np.random.uniform(1.2, 1.8)
            freq = 2 * math.pi / wavelength
            angle = np.random.uniform(0, 2*math.pi)
            proj = X * math.cos(angle) + Y * math.sin(angle)
            elevation = 0.5 * (np.sin(freq * proj) + 1.0) * difficulty * 0.55 # max ~2.7 cm

        elif mode == "slopes":
            # Gentle incline/decline ramp in random direction
            angle = np.random.uniform(0, 2*math.pi)
            grade = np.random.uniform(0.02, 0.04) * difficulty # 2% to 4% slope
            ramp = np.clip(((X - center_x) * math.cos(angle) + (Y - center_y) * math.sin(angle)) * grade, -0.02, 0.02)
            elevation = (ramp + 0.02) / self.hfield_max_z

        elif mode == "mixed":
            # Gentle waves combined with bumps
            w_wave = 0.5 * (np.sin(3.0 * X) * np.cos(2.5 * Y) + 1.0)
            w_noise = 0.5 * (np.sin(6.0 * X) * np.sin(5.5 * Y) + 1.0)
            elevation = (0.6 * w_wave + 0.4 * w_noise) * difficulty * 0.50

        else: # flat
            elevation = np.zeros((nrow, ncol), dtype=np.float32)

        final_elevation = np.clip(blend * elevation, 0.0, 1.0)
        self.model.hfield_data[:] = final_elevation.flatten()

        # Dynamically update the visual PBR material of the floor to match the active terrain
        mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"material_{mode}")
        if mat_id != -1 and self.floor_geom_id != -1:
            self.model.geom_matid[self.floor_geom_id] = mat_id

    def compute_cpg(self, phi: float, vx: float, vy: float, wz: float) -> np.ndarray:
        """Terrain-adaptive CPG with natural opened-outward stance and planar foot sole equilibrium."""
        speed = math.sqrt(vx*vx + vy*vy) + abs(wz) * 0.1
        # Natural opened-outward base posture matching walking stance width (15.3 cm)
        base_hr = 0.070
        base_hy = 0.035
        if speed < 0.025:
            # Planar flat sole standing: natural upright legs, straight vertical torso, flat soles
            return np.array([
                -base_hy,  base_hr, -0.08, 0.28, -0.20, -base_hr,
                 base_hy, -base_hr, -0.08, 0.28, -0.20,  base_hr
            ], dtype=np.float32)

        sin_p = math.sin(phi)
        swing_l = max(0.0, sin_p)
        swing_r = max(0.0, -sin_p)

        # 1. Pitch & Knee Kinematics: Natural Human Stride Length & Pitch CoM Equilibrium
        if vx >= 0.0:
            nom_pitch = -0.32 - 0.06 * math.tanh(2.5 * vx)
            # Natural human step reach: balanced hip excursion up to 0.22 rad (~13°)
            step_p = np.clip(0.85 * vx, 0.0, 0.22)
            clearance_pitch = -0.04 * math.tanh(2.5 * vx)
            knee_nom = 0.52 + 0.02 * math.tanh(2.5 * vx)
            knee_a = np.clip(0.42 + 0.14 * abs(vx) + 0.20 * abs(vy), 0.40, 0.52)
        else:
            # Human backward walking: upright torso (-0.22 rad), smooth backward reach
            nom_pitch = -0.22 + 0.12 * vx
            step_p = np.clip(0.60 * vx, -0.15, 0.0)
            clearance_pitch = 0.02 * abs(vx)
            knee_nom = 0.50
            knee_a = np.clip(0.38 + 0.10 * abs(vx), 0.35, 0.48)

        turn_pitch = np.clip(0.12 * wz, -0.07, 0.07)
        # Dynamic knee lift for smooth ground clearance
        l_toe_lift = clearance_pitch * swing_l
        r_toe_lift = clearance_pitch * swing_r

        lk = (knee_nom - 0.04 * swing_r) + knee_a * swing_l
        rk = (knee_nom - 0.04 * swing_l) + knee_a * swing_r

        lhp = nom_pitch - step_p * sin_p + turn_pitch * sin_p
        rhp = nom_pitch + step_p * sin_p - turn_pitch * sin_p

        # Foot pitch keeps sole parallel to ground during stance, with slight toe lift in swing
        lap = -(lhp + lk) + l_toe_lift
        rap = -(rhp + rk) + r_toe_lift

        # 2. Smooth Lateral Sway: Stabilizes CoM over Stance Foot without Excessive Wobble
        sway = -(0.034 + 0.010 * abs(vx) + 0.020 * abs(vy)) * sin_p
        if vy >= 0.0:
            lat_left  = np.clip( 1.35 * vy, 0.0, 0.20) * swing_l
            lat_right = np.clip( 0.45 * vy, 0.0, 0.05) * swing_r
        else:
            lat_left  = np.clip( 0.45 * vy, -0.05, 0.0) * swing_l
            lat_right = np.clip( 1.35 * vy, -0.20, 0.0) * swing_r

        lhr = np.clip( base_hr + sway + lat_left,  -0.08, 0.28)
        rhr = np.clip(-base_hr + sway + lat_right, -0.28, 0.08)
        lar = -lhr
        rar = -rhr

        # 3. Pure In-Place Yaw Rotation Stepping with natural outward base
        yaw_step = np.clip(0.32 * wz, -0.20, 0.20)
        lhy = -base_hy + yaw_step * swing_l
        rhy =  base_hy + yaw_step * swing_r

        return np.array([lhy, lhr, lhp, lk, lap, lar, rhy, rhr, rhp, rk, rap, rar], dtype=np.float32)

    def set_command(self, vx: float, vy: float, wz: float):
        self.command = np.array([vx, vy, wz], dtype=np.float32)

    def _sample_command(self):
        """Domain randomization across modes with terrain challenges."""
        mode = np.random.choice([
            "forward", "backward", "strafe_left", "strafe_right", 
            "turn_in_place_ccw", "turn_in_place_cw", "combined", "stop"
        ], p=[0.22, 0.20, 0.12, 0.14, 0.10, 0.10, 0.04, 0.08])
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
        else:
            self.command = np.zeros(3, dtype=np.float32)
            
        self.command_duration = np.random.randint(150, 300)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Procedural terrain generation on reset during training
        if self.training:
            self.generate_terrain(mode=self.terrain_type, difficulty=self.terrain_difficulty)

        self.phase = np.random.uniform(0.0, 2.0 * math.pi) if self.training else 0.0
        self.step_counter = 0
        self.push_steps = 0
        self.target_yaw = 0.0
        if self.training:
            self._sample_command()
            self.target_v = self.command.copy()
        else:
            self.target_v = np.zeros(3, dtype=np.float32)

        init_cpg = self.compute_cpg(self.phase, *self.target_v)
        for i, qpos_adr in enumerate(self.leg_qpos_indices):
            noise = np.random.uniform(-0.015, 0.015) if self.training else 0.0
            self.data.qpos[qpos_adr] = init_cpg[i] + noise

        for act_id, (qpos_adr, val) in self.upper_act_dict.items():
            self.data.qpos[qpos_adr] = val
            self.data.ctrl[act_id] = val

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.initial_root_z
        self.data.qpos[3] = 1.0
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = 0.0
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
        local_angvel = R.T @ self.data.qvel[3:6]
        local_linvel = R.T @ self.data.qvel[0:3]

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
            local_angvel * 0.25,                                         # 3
            local_linvel * 2.0,                                          # 3
            rel_joint_pos,                                               # 12
            joint_vel * 0.05,                                            # 12
            self.prev_action,                                            # 12
            np.array([sin_phase, cos_phase], dtype=np.float32),          # 2
            np.array([float(r_contact), float(l_contact)], dtype=np.float32), # 2
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

            if np.random.rand() < 0.005 and self.push_steps == 0:
                self.push_steps = np.random.randint(5, 12)
                self.push_force[:2] = np.random.uniform(-2.5, 2.5, size=2)

        max_dv = 0.5 * self.dt
        dv = np.clip(self.command - self.target_v, -max_dv, max_dv)
        self.target_v += dv

        cmd_speed = math.sqrt(self.target_v[0]**2 + self.target_v[1]**2) + abs(self.target_v[2]) * 0.1
        if cmd_speed > 0.02:
            self.phase = (self.phase + 2.0 * math.pi * self.gait_frequency * self.dt) % (2.0 * math.pi)

        cpg_targets = self.compute_cpg(self.phase, *self.target_v)
        # Smooth velocity gating: eliminate residual policy offsets when standing still
        # so both foot soles rest 100% planar and flat against the ground
        action_weight = min(1.0, max(0.0, (cmd_speed - 0.015) / 0.025))
        final_targets = cpg_targets + (self.action_scale * action_weight) * action

        # ============================================================
        # ACTIVE IMU TORSO BALANCE & CORRECTION REFLEX
        # IMU coordinate frame is located at torso site: pos="0.037 -0.007 0.424"
        # Corrects pitch (forward/back) and roll (lateral) when perturbed
        # Zero correction when unperturbed so foot soles remain 100% planar flat.
        # ============================================================
        q = self.data.qpos[3:7]
        w, x, y, z = q
        curr_pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
        curr_roll = math.atan2(float(2.0 * (w * x + y * z)), float(1.0 - 2.0 * (x * x + y * y)))

        nom_pitch = -0.025  # ~ -1.4 deg natural upright pitch
        target_pitch = nom_pitch + (0.06 * math.tanh(2.5 * self.target_v[0]) if self.target_v[0] > 0 else -0.03 * math.tanh(2.5 * abs(self.target_v[0])))
        p_err = curr_pitch - target_pitch
        p_rate = float(self.data.qvel[4])
        r_err = curr_roll
        r_rate = float(self.data.qvel[3])

        corr_pa = float(np.clip(0.50 * p_err + 0.04 * p_rate, -0.15, 0.15))
        corr_ph = float(np.clip(-0.30 * p_err - 0.02 * p_rate, -0.15, 0.15))
        corr_ra = float(np.clip(0.35 * r_err + 0.03 * r_rate, -0.10, 0.10))
        corr_rh = float(np.clip(-0.25 * r_err - 0.02 * r_rate, -0.10, 0.10))

        # Apply active IMU corrections to pitch (hips + ankles) and roll (hips + ankles)
        final_targets[2] += corr_ph   # left_hip_pitch
        final_targets[4] += corr_pa   # left_ankle_pitch
        final_targets[8] += corr_ph   # right_hip_pitch
        final_targets[10] += corr_pa  # right_ankle_pitch

        final_targets[1] += corr_rh   # left_hip_roll
        final_targets[5] += corr_ra   # left_ankle_roll
        final_targets[7] += corr_rh   # right_hip_roll
        final_targets[11] += corr_ra  # right_ankle_roll

        for i, act_id in enumerate(self.leg_act_indices):
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(final_targets[i], ctrl_min, ctrl_max)

        for act_id, (_, target) in self.upper_act_dict.items():
            self.data.ctrl[act_id] = target

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
            "rel_torso_z": torso_z - terrain_z
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

        # 1. Forward / Backward Velocity Tracking & Human-Scale Stride Propulsion
        err_vx = local_linvel[0] - self.target_v[0]
        r_vx = 15.0 * math.exp(-err_vx**2 / 0.010)
        r_fwd = 0.0
        p_stuck_x = 0.0
        r_fwd_prog = 0.0
        if self.target_v[0] > 0.03:
            if local_linvel[0] > 0.04:
                r_fwd = 45.0 * min(local_linvel[0], self.target_v[0])
                r_fwd_prog = 30.0 * min(local_linvel[0], self.target_v[0])
            else:
                p_stuck_x = 40.0 * (self.target_v[0] - max(0.0, local_linvel[0]))
        elif self.target_v[0] < -0.03:
            if local_linvel[0] < -0.03:
                r_fwd = 45.0 * min(-local_linvel[0], -self.target_v[0])
                r_fwd_prog = 30.0 * min(-local_linvel[0], -self.target_v[0])
            else:
                p_stuck_x = 40.0 * (-self.target_v[0] - max(-0.02, -local_linvel[0]))

        # 2. Lateral Strafe Velocity Tracking
        err_vy = local_linvel[1] - self.target_v[1]
        r_vy = 15.0 * math.exp(-err_vy**2 / 0.010)
        r_lat = 0.0
        p_stuck_y = 0.0
        if self.target_v[1] > 0.03:
            if local_linvel[1] > 0.02:
                r_lat = 25.0 * min(local_linvel[1], self.target_v[1])
            else:
                p_stuck_y = 20.0 * (self.target_v[1] - max(0.0, local_linvel[1]))
        elif self.target_v[1] < -0.03:
            if local_linvel[1] < -0.02:
                r_lat = 35.0 * min(-local_linvel[1], -self.target_v[1])
            else:
                p_stuck_y = 30.0 * (-self.target_v[1] - max(-0.02, -local_linvel[1]))

        # 3. Yaw Angular Velocity Tracking
        err_wz = local_angvel[2] - self.target_v[2]
        r_wz = 15.0 * math.exp(-err_wz**2 / 0.015)
        r_rot = 0.0
        p_stuck_wz = 0.0
        if abs(self.target_v[2]) > 0.08:
            if abs(local_angvel[2]) > 0.04 and (local_angvel[2] * self.target_v[2] > 0):
                r_rot = 25.0 * min(abs(local_angvel[2]), abs(self.target_v[2]))
            else:
                p_stuck_wz = 20.0 * abs(self.target_v[2])

        # 4. Strict Zero-Drift Penalties
        p_lat_drift = 40.0 * (local_linvel[1]**2) if abs(self.target_v[1]) < 0.02 else 0.0
        p_fwd_drift = 40.0 * (local_linvel[0]**2) if abs(self.target_v[0]) < 0.02 else 0.0
        p_yaw_drift = 35.0 * (local_angvel[2]**2) if abs(self.target_v[2]) < 0.02 else 0.0

        # 5. Heading Lock
        current_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        r_heading = 0.0
        p_heading = 0.0
        if abs(self.target_v[2]) < 0.02:
            yaw_diff = (current_yaw - self.target_yaw + math.pi) % (2 * math.pi) - math.pi
            r_heading = 8.0 * math.exp(-yaw_diff**2 / 0.02)
            p_heading = 30.0 * (yaw_diff**2)
        else:
            self.target_yaw = current_yaw

        p_turn_drift = 0.0
        if abs(self.target_v[2]) > 0.08 and abs(self.target_v[0]) < 0.03 and abs(self.target_v[1]) < 0.03:
            p_turn_drift = 35.0 * (local_linvel[0]**2 + local_linvel[1]**2)

        # 6. Upright Posture & Relative Terrain Height Maintenance
        r_upright = 8.0 * math.exp(-(1.0 + projected_gravity[2])**2 / 0.005)
        p_tilt = 16.0 * (projected_gravity[0]**2 + projected_gravity[1]**2)

        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        terrain_z = self.get_terrain_height(torso_x, torso_y)
        rel_torso_z = torso_z - terrain_z

        r_height = 8.0 * math.exp(-((rel_torso_z - self.nominal_torso_height)**2) / 0.005)
        p_sag = 25.0 * (max(0.0, 0.35 - rel_torso_z)**2)

        # 7. Foot Clearance on Terrain
        r_contact, l_contact, _ = self._check_contacts()
        sin_phase = math.sin(self.phase)
        lf_z = self.data.site_xpos[self.left_foot_site_id, 2]
        rf_z = self.data.site_xpos[self.right_foot_site_id, 2]
        lf_rel_z = lf_z - self.get_terrain_height(self.data.site_xpos[self.left_foot_site_id, 0], self.data.site_xpos[self.left_foot_site_id, 1])
        rf_rel_z = rf_z - self.get_terrain_height(self.data.site_xpos[self.right_foot_site_id, 0], self.data.site_xpos[self.right_foot_site_id, 1])

        r_clearance = 0.0
        cmd_speed = math.sqrt(self.target_v[0]**2 + self.target_v[1]**2)
        if cmd_speed > 0.03:
            if sin_phase > 0.35:
                r_clearance += 4.0 * min(max(0.0, (lf_rel_z - 0.008) / 0.040), 1.0)
            elif sin_phase < -0.35:
                r_clearance += 4.0 * min(max(0.0, (rf_rel_z - 0.008) / 0.040), 1.0)

        # Foot Slip Penalty
        lf_vel = np.zeros(6)
        rf_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.left_foot_site_id, lf_vel, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.right_foot_site_id, rf_vel, 0)
        lf_vxy2 = lf_vel[3]**2 + lf_vel[4]**2
        rf_vxy2 = rf_vel[3]**2 + rf_vel[4]**2
        p_slip = 0.4 * (lf_vxy2 * float(l_contact) + rf_vxy2 * float(r_contact))

        # 8. Stance Foot Flatness (Planar Ground Contact & Anti-Wobble)
        rf_mat = self.data.xmat[self.r_foot_bid].reshape(3, 3)
        lf_mat = self.data.xmat[self.l_foot_bid].reshape(3, 3)
        # Deviation of sole normal from world vertical Z
        rf_tilt2 = rf_mat[0, 2]**2 + rf_mat[1, 2]**2
        lf_tilt2 = lf_mat[0, 2]**2 + lf_mat[1, 2]**2
        stance_tilt = rf_tilt2 * float(r_contact) + lf_tilt2 * float(l_contact)
        r_flat_foot = 10.0 * math.exp(-stance_tilt / 0.003)
        p_foot_tilt = 35.0 * stance_tilt

        # Anti-Wobble: Strongly penalize rapid lateral roll oscillations of the torso
        p_roll_wobble = 2.5 * (local_angvel[0]**2)

        # 9. Smoothness & Energy
        p_smoothness = 0.02 * np.sum(np.square(action - self.prev_action))
        p_torque = 1e-4 * np.sum(np.square(self.data.qfrc_actuator[self.leg_qvel_indices]))
        p_gyro = 0.2 * np.sum(np.square(local_angvel))

        # 10. Survival Bonus
        r_healthy = 4.0

        total_reward = (
            r_vx + r_vy + r_wz +
            r_fwd + r_lat + r_rot +
            r_fwd_prog +
            r_heading +
            r_upright + r_height +
            r_clearance +
            r_flat_foot +
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
            p_foot_tilt -
            p_roll_wobble -
            p_gyro -
            p_smoothness -
            p_torque -
            p_slip -
            p_sag
        )

        return total_reward

    def _is_terminated(self):
        torso_x = self.data.site_xpos[self.torso_site_id, 0]
        torso_y = self.data.site_xpos[self.torso_site_id, 1]
        torso_z = self.data.site_xpos[self.torso_site_id, 2]
        terrain_z = self.get_terrain_height(torso_x, torso_y)
        rel_torso_z = torso_z - terrain_z

        # Fall detection relative to local terrain height
        if rel_torso_z < 0.20 or rel_torso_z > 0.55:
            return True

        # Fall detection by tilt (> 60 deg)
        q = self.data.qpos[3:7]
        w, x, y, z = q
        gz = -(1.0 - 2.0*(x**2 + y**2))
        if gz > -0.50:
            return True

        return False
