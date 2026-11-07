"""
collect_dataset.py
------------------
Collects physical state transitions (s_t, a_t, s_{t+1}) from MuJoCo
around the MoCap walking trajectory using Reference State Initialization (RSI).
This dataset trains the SuperMimic Differentiable World Model.
"""

import os
import math
import numpy as np
import mujoco
from supermimic.mocap_dataset import MocapDataset, LEG_QPOS_ADR, LEG_QVEL_ADR, LEG_ACT_ID

class TransitionCollector:
    def __init__(
        self,
        xml_path: str = "barelangS_terrain_mujoco.xml",
        mocap_path: str = "data/mocap/barelang_mocap_walk.npz"
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.mocap = MocapDataset(mocap_path)

        # Upper body passive posture targets
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

        # Simulation parameters
        # MuJoCo timestep is typically 0.002s (500Hz)
        # Control step matches 60Hz MoCap (dt ~ 0.01667s, nsubsteps = 8)
        self.sim_dt = self.model.opt.timestep
        self.control_dt = 1.0 / 60.0
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))

    def _get_state(self) -> np.ndarray:
        """
        Extracts 34-dim state vector:
          [0:3]   Projected gravity vector (3)
          [3]     Root Z height (1)
          [4:7]   Base linear velocity in base frame (3)
          [7:10]  Base angular velocity in base frame (3)
          [10:22] 12 leg joint angles (12)
          [22:34] 12 leg joint velocities (12)
        """
        # Base orientation quaternion [w, x, y, z]
        w, x, y, z = self.data.qpos[3:7]
        # Rotation matrix R (world to base)
        R = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
            [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)]
        ], dtype=np.float32)

        # Projected gravity
        proj_grav = -R[2, :]  # (3,)

        # Root z
        root_z = np.array([self.data.qpos[2]], dtype=np.float32)

        # Base velocities rotated into base frame
        lin_vel_world = self.data.qvel[0:3]
        ang_vel_world = self.data.qvel[3:6]
        lin_vel_base = R.T @ lin_vel_world
        ang_vel_base = R.T @ ang_vel_world

        # Leg joints
        leg_qpos = self.data.qpos[LEG_QPOS_ADR]
        leg_qvel = self.data.qvel[LEG_QVEL_ADR]

        state = np.concatenate([
            proj_grav,
            root_z,
            lin_vel_base,
            ang_vel_base,
            leg_qpos,
            leg_qvel
        ], dtype=np.float32)

        return state

    def reset_to_mocap(self, frame_idx: int, noise_std: float = 0.02):
        """Initializes the simulation near a reference MoCap frame."""
        ref_qpos, ref_qvel = self.mocap.get_reference_pose(frame_idx)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = ref_qpos
        self.data.qvel[:] = ref_qvel

        # Add small exploration noise
        if noise_std > 0:
            self.data.qpos[LEG_QPOS_ADR] += np.random.normal(0, noise_std, size=12)
            self.data.qvel[LEG_QVEL_ADR] += np.random.normal(0, noise_std * 5.0, size=12)

        # Upper body
        for act_id, val in self.upper_act_dict.items():
            self.data.ctrl[act_id] = val

        mujoco.mj_forward(self.model, self.data)

    def collect_dataset(
        self,
        num_transitions: int = 50000,
        output_path: str = "data/mocap/world_model_transitions.npz"
    ):
        print(f"Collecting {num_transitions} transitions for Differentiable World Model...")

        states = np.zeros((num_transitions, 34), dtype=np.float32)
        actions = np.zeros((num_transitions, 12), dtype=np.float32)
        next_states = np.zeros((num_transitions, 34), dtype=np.float32)

        step_count = 0
        ep_len = 30  # short rollouts to stay near manifold
        cur_frame = self.mocap.sample_frame()

        while step_count < num_transitions:
            cur_frame = self.mocap.sample_frame()
            self.reset_to_mocap(cur_frame, noise_std=0.03)

            for _ in range(ep_len):
                if step_count >= num_transitions:
                    break

                s_t = self._get_state()

                # Action: reference tracking target + exploratory noise in [-1, 1]
                ref_targets = self.mocap.get_leg_target(cur_frame)
                action = np.random.uniform(-1.0, 1.0, size=12).astype(np.float32)

                # Convert action to PD position setpoint: target = ref + 0.3 * action
                pd_targets = ref_targets + 0.30 * action

                # Set actuator controls
                for i, act_id in enumerate(LEG_ACT_ID):
                    ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_id]
                    self.data.ctrl[act_id] = np.clip(pd_targets[i], ctrl_min, ctrl_max)

                for act_id, val in self.upper_act_dict.items():
                    self.data.ctrl[act_id] = val

                # Step sub-steps
                for _ in range(self.n_substeps):
                    mujoco.mj_step(self.model, self.data)

                s_next = self._get_state()

                states[step_count] = s_t
                actions[step_count] = action
                next_states[step_count] = s_next

                step_count += 1
                cur_frame = (cur_frame + 1) % self.mocap.n_frames

                # Check if robot fell completely (e.g. root_z < -0.15 or excessive tilt)
                if s_next[3] < -0.15 or s_next[2] < 0.3:
                    break

            if step_count % 10000 == 0:
                print(f"  Progress: {step_count}/{num_transitions} transitions ({step_count/num_transitions*100:.1f}%)")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            states=states,
            actions=actions,
            next_states=next_states
        )
        print(f"\n[OK] Transition dataset saved to: {output_path}")
        print(f"Dataset shape: states={states.shape}, actions={actions.shape}, next_states={next_states.shape}")

if __name__ == "__main__":
    collector = TransitionCollector()
    collector.collect_dataset(num_transitions=50000)

