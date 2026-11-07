#!/usr/bin/env python3
"""
view_mocap_barelang.py
----------------------
Interactive 3D Visualizer & Video Renderer for BarelangS Humanoid walking motion
retargeted from gold-standard human Motion Capture (MoCap).

Controls:
  [SPACE]      : Pause / Play
  [R]          : Reset to start
  [C]          : Toggle between full trajectory and single stride loop
  [1] / [2]    : Slow down / Speed up playback (0.25x, 0.5x, 1.0x, 2.0x)
  [RIGHT]      : Next frame (when paused)
  [LEFT]       : Previous frame (when paused)
  [ESC]        : Exit

Usage:
  # Interactive 3D Window (GLFW)
  python view_mocap_barelang.py
  python view_mocap_barelang.py --speed 0.5 --mode loop

  # Headless video rendering (if no display or want MP4)
  python view_mocap_barelang.py --render-video data/mocap/barelang_walk.mp4 --repeats 3
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco

KEY_SPACE = 32
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_R = 82
KEY_C = 67
KEY_1 = 49
KEY_2 = 50

class MocapViewer:
    def __init__(self, data_path: str, xml_path: str, initial_speed: float = 1.0, loop_cycle: bool = False):
        print(f"Loading retargeted MoCap from: {data_path}")
        data = np.load(data_path)
        self.qpos = data["qpos"]
        self.qvel = data["qvel"]
        self.fps = int(data["fps"])
        self.dt = float(data["dt"])
        self.n_frames = len(self.qpos)
        self.cycle_start = int(data.get("cycle_start", 18))
        self.cycle_end = int(data.get("cycle_end", 80))
        self.walking_speed = float(data.get("walking_speed", 0.71))
        self.stride_len = float(data.get("stride_length", 0.73))

        print(f"Loading MuJoCo model from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.current_frame = 0
        self.is_paused = False
        self.playback_speed = initial_speed
        self.loop_cycle = loop_cycle
        self.speeds = [0.25, 0.5, 1.0, 1.5, 2.0]

    def key_callback(self, keycode: int):
        if keycode == KEY_SPACE:
            self.is_paused = not self.is_paused
            print(f"\n[{'PAUSED' if self.is_paused else 'PLAYING'}]")
        elif keycode == KEY_R:
            self.current_frame = self.cycle_start if self.loop_cycle else 0
            print("\n[RESET] Returned to starting frame")
        elif keycode == KEY_C:
            self.loop_cycle = not self.loop_cycle
            mode_str = "Single Stride Cycle (Seamless Loop)" if self.loop_cycle else "Full Trajectory (154 frames)"
            print(f"\n[MODE] Switched to: {mode_str}")
            if self.loop_cycle and self.current_frame < self.cycle_start:
                self.current_frame = self.cycle_start
        elif keycode == KEY_1:
            idx = max(0, self.speeds.index(self.playback_speed) - 1) if self.playback_speed in self.speeds else 1
            self.playback_speed = self.speeds[idx]
            print(f"\n[SPEED] {self.playback_speed:.2f}x")
        elif keycode == KEY_2:
            idx = min(len(self.speeds) - 1, self.speeds.index(self.playback_speed) + 1) if self.playback_speed in self.speeds else 2
            self.playback_speed = self.speeds[idx]
            print(f"\n[SPEED] {self.playback_speed:.2f}x")
        elif keycode == KEY_RIGHT:
            self.advance_frame(1)
            print(f"\n[FRAME] Step -> {self.current_frame}/{self.n_frames}")
        elif keycode == KEY_LEFT:
            self.advance_frame(-1)
            print(f"\n[FRAME] Step <- {self.current_frame}/{self.n_frames}")

    def advance_frame(self, delta: int = 1):
        if self.loop_cycle:
            cycle_len = self.cycle_end - self.cycle_start
            rel = (self.current_frame - self.cycle_start + delta) % cycle_len
            self.current_frame = self.cycle_start + rel
        else:
            self.current_frame = (self.current_frame + delta) % self.n_frames

    def render_to_video(self, output_path: str, repeats: int = 1, width: int = 640, height: int = 480):
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) is required to render video. Install with: pip install opencv-python")
            return

        print(f"\nRendering video to {output_path} ({width}x{height} @ {self.fps}fps)...")
        renderer = mujoco.Renderer(self.model, height, width)
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
        writer = cv2.VideoWriter(output_path, fourcc, float(self.fps), (width, height))

        total_frames = self.n_frames * repeats
        count = 0
        for rep in range(repeats):
            for f in range(self.n_frames):
                self.data.qpos[:] = self.qpos[f]
                self.data.qvel[:] = self.qvel[f]
                mujoco.mj_forward(self.model, self.data)
                renderer.update_scene(self.data, camera=camera)
                rgb = renderer.render()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                count += 1
                if count % 30 == 0 or count == total_frames:
                    print(f"\rRendered {count}/{total_frames} frames ({count/total_frames*100:.1f}%)", end="", flush=True)

        writer.release()
        print(f"\n[OK] Video saved successfully to: {output_path}")

    def run_gui(self):
        try:
            import mujoco.viewer
        except ImportError:
            print("[ERROR] mujoco.viewer requires OpenGL/GLFW. Use --render-video instead.")
            return

        print("=" * 75)
        print(" BARELANG-S MOCAP WALKING VISUALIZER (SuperMimic Reference Trajectory)")
        print("=" * 75)
        print("  Controls:")
        print("    [SPACE]   : Pause / Play")
        print("    [R]       : Reset to start")
        print("    [C]       : Toggle Full Path vs Stride Loop")
        print("    [1] / [2] : Slower / Faster playback")
        print("    [LEFT/RIGHT] Arrow : Step 1 frame forward/backward")
        print("=" * 75)

        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
                torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_2025")
                if torso_bid != -1:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = torso_bid
                    viewer.cam.distance = 2.0
                    viewer.cam.elevation = -12.0
                    viewer.cam.azimuth = 115.0

                last_hud = 0.0

                while viewer.is_running():
                    loop_start = time.time()

                    # Set kinematic pose
                    self.data.qpos[:] = self.qpos[self.current_frame]
                    self.data.qvel[:] = self.qvel[self.current_frame]
                    mujoco.mj_forward(self.model, self.data)

                    viewer.sync()

                    # Terminal HUD
                    now = time.time()
                    if now - last_hud > 0.12:
                        last_hud = now
                        r_knee_deg = np.degrees(self.qpos[self.current_frame, 18])
                        l_knee_deg = np.degrees(self.qpos[self.current_frame, 24])
                        r_hip_deg = np.degrees(self.qpos[self.current_frame, 17])
                        l_hip_deg = np.degrees(self.qpos[self.current_frame, 23])
                        mode_lbl = "LOOP" if self.loop_cycle else "FULL"
                        state_lbl = "PAUSE" if self.is_paused else f"{self.playback_speed:.2f}x"

                        hud = (
                            f"\r[{mode_lbl}|{state_lbl}] Frame: {self.current_frame:03d}/{self.n_frames} | "
                            f"Pos X: {self.qpos[self.current_frame, 0]:.2f}m | "
                            f"Knee: R={r_knee_deg:4.1f}° L={l_knee_deg:4.1f}° | "
                            f"Hip Pitch: R={r_hip_deg:4.1f}° L={l_hip_deg:4.1f}°"
                        )
                        sys.stdout.write(hud)
                        sys.stdout.flush()

                    # Advance frame if playing
                    if not self.is_paused:
                        self.advance_frame(1)

                    # Maintain playback timing
                    frame_interval = self.dt / self.playback_speed
                    elapsed = time.time() - loop_start
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)

        except Exception as e:
            print(f"\n[NOTICE] Could not open GUI viewer ({e}).")
            print("Tip: If you are running headless or via SSH, render a video instead:")
            print("  python view_mocap_barelang.py --render-video data/mocap/barelang_walk.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View retargeted MoCap walking motion on BarelangS")
    parser.add_argument("--data", type=str, default="data/mocap/barelang_mocap_walk.npz", help="Path to retargeted mocap .npz")
    parser.add_argument("--xml", type=str, default="barelangS_terrain_mujoco.xml", help="Path to BarelangS xml")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed (default: 1.0)")
    parser.add_argument("--mode", type=str, choices=["full", "loop"], default="loop", help="Playback mode: full or loop")
    parser.add_argument("--render-video", type=str, default=None, help="If provided, renders MP4 video to this path")
    parser.add_argument("--repeats", type=int, default=2, help="Number of times to loop sequence in rendered video")
    args = parser.parse_args()

    viewer = MocapViewer(
        data_path=args.data,
        xml_path=args.xml,
        initial_speed=args.speed,
        loop_cycle=(args.mode == "loop")
    )

    if args.render_video:
        viewer.render_to_video(args.render_video, repeats=args.repeats)
    else:
        viewer.run_gui()
