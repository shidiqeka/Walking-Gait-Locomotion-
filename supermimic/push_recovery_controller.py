"""
push_recovery_controller.py
---------------------------
Capture-Point and Bio-Mechanical Stepping Recovery Controller for BarelangS.
Stabilizes the robot against external disturbances (pushes/collisions) using:
  1. Ankle Stance Compliance (Ankle Strategy)
  2. Capture-Point Dynamic Stepping Modification (Stepping Strategy)
  3. Dynamic Inertial Arm Damping (Angular Momentum Regulation)
  4. Phase Acceleration (Early Touchdown)
"""

import math
import numpy as np

class PushRecoveryController:
    def __init__(self, leg_length: float = 0.38):
        self.leg_length = leg_length
        # Natural inverted pendulum frequency: omega_0 = sqrt(g / L)
        self.omega_0 = math.sqrt(9.81 / leg_length)  # ~5.08 rad/s
        self.tau_pred = 1.0 / self.omega_0           # ~0.197 s

        # Disturbance states
        self.pitch = 0.0
        self.roll = 0.0
        self.pitch_rate = 0.0
        self.roll_rate = 0.0
        self.lin_vx = 0.0
        self.lin_vy = 0.0

        # Filtered rates
        self.filtered_pitch_rate = 0.0
        self.filtered_roll_rate = 0.0

        # Recovery status
        self.is_disturbed = False
        self.disturbance_magnitude = 0.0

    def reset(self):
        """Resets all internal filter states and disturbance estimates."""
        self.pitch = 0.0
        self.roll = 0.0
        self.pitch_rate = 0.0
        self.roll_rate = 0.0
        self.lin_vx = 0.0
        self.lin_vy = 0.0
        self.filtered_pitch_rate = 0.0
        self.filtered_roll_rate = 0.0
        self.is_disturbed = False
        self.disturbance_magnitude = 0.0

    def update_imu(self, quat_wxyz: np.ndarray, gyro_xyz: np.ndarray, lin_vel_base: np.ndarray, dt: float = 0.01667):
        """
        Updates internal state estimates from IMU and velocity measurements.
        quat_wxyz: [w, x, y, z]
        gyro_xyz: [wx, wy, wz] in base frame
        lin_vel_base: [vx, vy, vz] in base frame
        """
        w, x, y, z = quat_wxyz

        # Extract pitch and roll (Euler angles in ZYX convention)
        # pitch: rotation around base Y axis
        sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        self.pitch = math.asin(sin_pitch)

        # roll: rotation around base X axis
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        self.roll = math.atan2(sin_roll, cos_roll)

        # Gyro rates (wy = pitch rate, wx = roll rate)
        alpha = 0.85
        self.filtered_pitch_rate = alpha * self.filtered_pitch_rate + (1.0 - alpha) * gyro_xyz[1]
        self.filtered_roll_rate  = alpha * self.filtered_roll_rate  + (1.0 - alpha) * gyro_xyz[0]
        self.pitch_rate = self.filtered_pitch_rate
        self.roll_rate = self.filtered_roll_rate

        self.lin_vx = lin_vel_base[0]
        self.lin_vy = lin_vel_base[1]

        # Total disturbance magnitude
        tilt_norm = math.sqrt(self.pitch**2 + self.roll**2)
        rate_norm = math.sqrt(self.pitch_rate**2 + self.roll_rate**2)
        self.disturbance_magnitude = tilt_norm + 0.15 * rate_norm
        self.is_disturbed = self.disturbance_magnitude > 0.045  # ~2.5 deg threshold

    def compute_recovery_offsets(self, swing_leg: str = "right") -> dict:
        """
        Computes additive joint target offsets for push recovery.
        swing_leg: 'right' or 'left' (the foot currently in flight)
        Returns:
          dict with:
            'leg_offsets': (12,) array in canonical leg order:
               [L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle_pitch, L_ankle_roll,
                R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle_pitch, R_ankle_roll]
            'arm_pitch_offset': float
            'arm_roll_offset': float
            'phase_acceleration': float multiplier (e.g. 1.0 to 1.6)
        """
        # 1. Capture Point predicted tilt (future state at tau_pred)
        pred_pitch = self.pitch + self.tau_pred * self.pitch_rate
        pred_roll  = self.roll  + self.tau_pred * self.roll_rate

        leg_offsets = np.zeros(12, dtype=np.float32)

        # Canonical indexing:
        # Left leg:  0=hip_yaw, 1=hip_roll, 2=hip_pitch, 3=knee, 4=ankle_p, 5=ankle_r
        # Right leg: 6=hip_yaw, 7=hip_roll, 8=hip_pitch, 9=knee, 10=ankle_p, 11=ankle_r

        # --- A. ANKLE STRATEGY (Stance foot compliance) ---
        # Resisting tilt through ankle restoring moments
        ankle_p_corr = float(np.clip(-0.55 * pred_pitch, -0.30, 0.30))
        ankle_r_corr = float(np.clip(-0.40 * pred_roll,  -0.25, 0.25))

        leg_offsets[4]  += ankle_p_corr  # Left ankle pitch
        leg_offsets[10] += ankle_p_corr  # Right ankle pitch
        leg_offsets[5]  += ankle_r_corr  # Left ankle roll
        leg_offsets[11] += ankle_r_corr  # Right ankle roll

        # --- B. STEPPING STRATEGY (Capture Point foot placement) ---
        # Adjust swing leg hip pitch to place the foot in the capture point direction
        # If falling forward (pred_pitch > 0), swing leg steps more forward (negative hip pitch)
        step_pitch_offset = float(np.clip(-0.85 * pred_pitch - 0.20 * self.lin_vx, -0.45, 0.45))
        knee_flex_offset  = float(np.clip(0.35 * abs(pred_pitch), 0.0, 0.30))

        # Lateral stepping: widen stance into lateral push direction
        step_roll_offset = float(np.clip(-0.60 * pred_roll - 0.25 * self.lin_vy, -0.35, 0.35))

        if swing_leg == "right":
            # Right leg is swing leg
            leg_offsets[8] += step_pitch_offset  # R hip pitch
            leg_offsets[9] += knee_flex_offset   # R knee
            leg_offsets[7] += step_roll_offset   # R hip roll
            # Stance leg (left) pushes slightly backward to propel CoM
            leg_offsets[2] -= 0.20 * step_pitch_offset
        else:
            # Left leg is swing leg
            leg_offsets[2] += step_pitch_offset  # L hip pitch
            leg_offsets[3] += knee_flex_offset   # L knee
            leg_offsets[1] += step_roll_offset   # L hip roll
            # Stance leg (right) pushes slightly backward
            leg_offsets[8] -= 0.20 * step_pitch_offset

        # --- C. DYNAMIC INERTIAL ARM DAMPING ---
        # Counter-torque: fling arms opposite to torso angular acceleration
        arm_pitch_offset = float(np.clip(-1.4 * pred_pitch - 0.12 * self.pitch_rate, -0.55, 0.55))
        arm_roll_offset  = float(np.clip(0.50 * abs(pred_roll), 0.0, 0.35))

        # --- D. EARLY TOUCHDOWN (Phase Acceleration) ---
        # If heavily disturbed, accelerate gait to touch down quickly
        if abs(pred_pitch) > 0.06 or abs(pred_roll) > 0.05:
            phase_accel = float(np.clip(1.0 + 3.5 * self.disturbance_magnitude, 1.0, 1.85))
        else:
            phase_accel = 1.0

        return {
            "leg_offsets": leg_offsets,
            "arm_pitch_offset": arm_pitch_offset,
            "arm_roll_offset": arm_roll_offset,
            "phase_acceleration": phase_accel,
            "pred_pitch": pred_pitch,
            "pred_roll": pred_roll,
        }

