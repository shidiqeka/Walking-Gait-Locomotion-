"""
evaluate_policy.py
------------------
Evaluates the trained SuperMimic policy in the full MuJoCo physics engine.
Tests closed-loop tracking of the MoCap walking trajectory.
Can render an MP4 video or run interactive GUI viewer.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import mujoco
from supermimic.mocap_dataset import MocapDataset, LEG_QPOS_ADR, LEG_QVEL_ADR, LEG_ACT_ID
from supermimic.policy import TrackingPolicy

class SuperMimicEvaluator:
    def __init__(
        self,
        policy_path: str = "models/supermimic_policy.pt",
        xml_path: str = "barelangS_terrain_mujoco.xml",
        mocap_path: str = "data/mocap/barelang_mocap_walk.npz",
        device: str = "cpu"
    ):
        self.device = device
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.mocap = MocapDataset(mocap_path, device=device)

        print(f"Loading SuperMimic policy from: {policy_path}")
        checkpoint = torch.load(policy_path, map_location=device, weights_only=True)
        self.policy = TrackingPolicy(state_dim=34, action_dim=12).to(device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()

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

        self.control_dt = 1.0 / 60.0
        self.sim_dt = self.model.opt.timestep
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))

    def _get_state(self) -> np.ndarray:
        w, x, y, z = self.data.qpos[3:7]
        R = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w)],
            [2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y)]
        ], dtype=np.float32)

        proj_grav = -R[2, :]
        root_z = np.array([self.data.qpos[2]], dtype=np.float32)
        lin_vel_base = R.T @ self.data.qvel[0:3]
        ang_vel_base = R.T @ self.data.qvel[3:6]
        leg_qpos = self.data.qpos[LEG_QPOS_ADR]
        leg_qvel = self.data.qvel[LEG_QVEL_ADR]

        return np.concatenate([
            proj_grav, root_z, lin_vel_base, ang_vel_base, leg_qpos, leg_qvel
        ], dtype=np.float32)

    def reset(self, start_frame: int = 18):
        ref_qpos, ref_qvel = self.mocap.get_reference_pose(start_frame)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = ref_qpos
        self.data.qvel[:] = ref_qvel
        for act_id, val in self.upper_act_dict.items():
            self.data.ctrl[act_id] = val
        mujoco.mj_forward(self.model, self.data)

    def render_video(self, output_path: str = "data/mocap/supermimic_walk_sim.mp4", max_steps: int = 300):
        import cv2
        print(f"\nEvaluating SuperMimic policy in MuJoCo physics (Rendering {max_steps} steps to {output_path})...")
        renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
        if torso_bid != -1:
            camera.trackbodyid = torso_bid
        camera.distance = 2.0
        camera.elevation = -12.0
        camera.azimuth = 115.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, 60.0, (640, 480))

        cur_frame = self.mocap.cycle_start
        self.reset(cur_frame)

        for step in range(max_steps):
            s_t = torch.tensor(self._get_state(), dtype=torch.float32, device=self.device).unsqueeze(0)
            ref_target = torch.tensor(self.mocap.get_leg_target(cur_frame), dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action = self.policy(s_t, ref_target).squeeze(0).cpu().numpy()

            pd_targets = self.mocap.get_leg_target(cur_frame) + 0.30 * action

            for i, act_id in enumerate(LEG_ACT_ID):
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[act_id]
                self.data.ctrl[act_id] = np.clip(pd_targets[i], ctrl_min, ctrl_max)

            for act_id, val in self.upper_act_dict.items():
                self.data.ctrl[act_id] = val

            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)

            cur_frame = self.mocap.cycle_start + (cur_frame - self.mocap.cycle_start + 1) % self.mocap.cycle_len

            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if step % 30 == 0:
                print(f"\rStep {step}/{max_steps} | Robot X: {self.data.qpos[0]:.2f}m", end="", flush=True)

        writer.release()
        print(f"\n[OK] Video saved: {output_path} (Final X: {self.data.qpos[0]:.2f}m)")

if __name__ == "__main__":
    evaluator = SuperMimicEvaluator()
    evaluator.render_video()

