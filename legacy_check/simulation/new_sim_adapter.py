"""
New Simulation Adapter for Emotion Pipeline

This adapter wraps the new locomotive-capable Spot simulation (spotBezierEnv)
and provides the SpotEnv-compatible interface required by the emotion pipeline.

Key features:
- Maps joint names from old convention (fl.hx, fl.hy, fl.kn) to new (motor_front_left_hip, etc.)
- Implements BezierStepper integration for smooth locomotion when emotions require movement
- Uses direct joint control for precise emotional poses and expressions
- Maintains overlay system and baseline pose system from old simulation
"""

import time
import numpy as np
import pybullet as p
from typing import Optional, Dict, Tuple
import sys
import os

# Add spotmicro to path
# Calculate absolute path to spotmicro directory
_current_file = os.path.abspath(__file__)
# From: spot_mini_mini/spot_bullet/src/emotion/integration/new_sim_adapter.py
# Go up: integration/ -> emotion/ -> src/ -> spot_bullet/ -> spot_mini_mini/
_emotion_dir = os.path.dirname(os.path.dirname(_current_file))  # emotion/
_spot_bullet_src_dir = os.path.dirname(_emotion_dir)  # src/
_spot_bullet_dir = os.path.dirname(_spot_bullet_src_dir)  # spot_bullet/
_spot_mini_dir = os.path.dirname(_spot_bullet_dir)  # spot_mini_mini/
_spotmicro_path = os.path.join(_spot_mini_dir, 'spotmicro')
# Also add spot_mini_mini to path so imports work
if _spot_mini_dir not in sys.path:
    sys.path.insert(0, _spot_mini_dir)
if _spotmicro_path not in sys.path:
    sys.path.insert(0, _spotmicro_path)

from spotmicro.OpenLoopSM.SpotOL import BezierStepper


class NewSimSpotAdapter:
    """
    Adapter that wraps spotBezierEnv/spotGymEnv to provide SpotEnv-compatible interface
    for the emotion pipeline.
    
    Uses hybrid approach:
    - BezierStepper for locomotion (retreat/approach movements)
    - Direct joint control for poses and expressions
    """
    
    def __init__(self, new_env):
        """
        Initialize adapter.
        
        Args:
            new_env: spotBezierEnv or spotGymEnv instance
        """
        self.env = new_env
        self.spot = new_env.spot
        self._pybullet_client = new_env._pybullet_client
        self.quadruped = new_env.spot.quadruped
        
        # Build joint name mapping (old -> new)
        self._build_joint_mapping()
        
        # Initialize BezierStepper for locomotion
        self.bezier_stepper = BezierStepper(
            dt=new_env._time_step,
            mode=0  # FWD mode (forward/backward only)
        )
        
        # Locomotion state
        self._use_locomotion = False
        self._locomotion_target_dy = 0.0
        self._locomotion_duration = 0.0
        self._locomotion_start_time = 0.0
        self._locomotion_active = False
        
        # Robot state tracking (from old SpotEnv)
        self.current_speed = 1.0
        self.current_distance = 1.0
        self.is_moving = False
        self.target_position = [0, 0, 0.4]
        
        # Safety monitoring
        self.safety_enabled = False
        self.max_roll_pitch = 0.5
        self.max_joint_velocity = 3.0
        self.safe_overlay_amplitude = 0.04
        self.safe_overlay_freq_range = (0.3, 2.0)
        self.last_safety_check_time = time.time()
        self.safety_check_interval = 0.1
        
        # Global motion intensity scalar
        self.motion_intensity: float = 0.3
        
        # Safe modulation subspace
        self.safe_joint_deltas = {
            'fl.hy': 0.05, 'fr.hy': 0.05, 'hl.hy': 0.05, 'hr.hy': 0.05,
            'fl.kn': 0.08, 'fr.kn': 0.08, 'hl.kn': 0.08, 'hr.kn': 0.08,
            'fl.hx': 0.04, 'fr.hx': 0.04, 'hl.hx': 0.04, 'hr.hx': 0.04,
        }
        
        # Baseline pose & dynamic overlay state
        self.baseline_pose_name: str = "NEUTRAL"
        self.baseline_joint_targets: Dict[str, float] = {}
        
        # Dynamic overlay configuration
        self.active_overlay: Optional[str] = None
        self.overlay_start_time: float = 0.0
        self.overlay_duration: float = 1.5
        self.overlay_params: Dict = {}
        
        # Define baseline poses - EXAGGERATED for clear visibility
        self.baseline_poses: Dict[str, Dict[str, float]] = {
            "NEUTRAL": {
                'fl.hx': 0.0, 'fl.hy': 0.6, 'fl.kn': -1.4,
                'fr.hx': 0.0, 'fr.hy': 0.6, 'fr.kn': -1.4,
                'hl.hx': 0.0, 'hl.hy': 0.6, 'hl.kn': -1.4,
                'hr.hx': 0.0, 'hr.hy': 0.6, 'hr.kn': -1.4,
            },
            "DE_ESCALATE": {
                # Clear crouch: higher hip (more extended), much more bent knees
                'fl.hx': 0.0, 'fl.hy': 1.0, 'fl.kn': -2.2,  # Much more crouched
                'fr.hx': 0.0, 'fr.hy': 1.0, 'fr.kn': -2.2,
                'hl.hx': 0.0, 'hl.hy': 1.1, 'hl.kn': -2.4,  # Hind legs even more crouched
                'hr.hx': 0.0, 'hr.hy': 1.1, 'hr.kn': -2.4,
            },
            "CHECK_IN": {
                # Forward lean: front legs lower, back legs higher
                'fl.hx': 0.0, 'fl.hy': 0.8, 'fl.kn': -1.8,  # Front legs lower
                'fr.hx': 0.0, 'fr.hy': 0.8, 'fr.kn': -1.8,
                'hl.hx': 0.0, 'hl.hy': 0.4, 'hl.kn': -1.0,  # Back legs straighter/higher
                'hr.hx': 0.0, 'hr.hy': 0.4, 'hr.kn': -1.0,
            },
            "ENGAGE": {
                # Tall and open: straighter legs, lower hip (more extended)
                'fl.hx': 0.0, 'fl.hy': 0.3, 'fl.kn': -0.8,  # Front legs very straight
                'fr.hx': 0.0, 'fr.hy': 0.3, 'fr.kn': -0.8,
                'hl.hx': 0.0, 'hl.hy': 0.8, 'hl.kn': -1.8,  # Back legs more extended
                'hr.hx': 0.0, 'hr.hy': 0.8, 'hr.kn': -1.8,
            },
            "CAUTION": {
                # Alert stance: slightly raised front, stable back
                'fl.hx': 0.0, 'fl.hy': 0.4, 'fl.kn': -1.0,  # Front legs straighter
                'fr.hx': 0.0, 'fr.hy': 0.4, 'fr.kn': -1.0,
                'hl.hx': 0.0, 'hl.hy': 0.7, 'hl.kn': -1.6,  # Back legs stable
                'hr.hx': 0.0, 'hr.hy': 0.7, 'hr.kn': -1.6,
            },
        }
        
        # Store spotId for compatibility
        self.spotId = self.quadruped
        
        # Joint targets for direct control
        self._joint_targets: Dict[str, float] = {}
        self._use_direct_control = True  # Default to direct control for poses
        
        print("[NewSimSpotAdapter] Initialized with BezierStepper locomotion support")
    
    def _build_joint_mapping(self):
        """Build mapping from old joint names to new motor indices."""
        # Old convention: fl.hx, fl.hy, fl.kn (front left hip x, hip y, knee)
        # New convention: motor_front_left_hip, motor_front_left_upper_leg, motor_front_left_lower_leg
        
        # Get all joint names from Spot
        self.joints: Dict[str, int] = {}
        self._old_to_new_mapping: Dict[str, str] = {}
        self._new_to_old_mapping: Dict[str, str] = {}
        
        # Map old names to new names
        mapping = {
            'fl.hx': 'motor_front_left_hip',
            'fl.hy': 'motor_front_left_upper_leg',
            'fl.kn': 'motor_front_left_lower_leg',
            'fr.hx': 'motor_front_right_hip',
            'fr.hy': 'motor_front_right_upper_leg',
            'fr.kn': 'motor_front_right_lower_leg',
            'hl.hx': 'motor_back_left_hip',
            'hl.hy': 'motor_back_left_upper_leg',
            'hl.kn': 'motor_back_left_lower_leg',
            'hr.hx': 'motor_back_right_hip',
            'hr.hy': 'motor_back_right_upper_leg',
            'hr.kn': 'motor_back_right_lower_leg',
        }
        
        # Build joint name to index mapping from Spot's internal structure
        if hasattr(self.spot, '_joint_name_to_id') and self.spot._joint_name_to_id:
            for old_name, new_name in mapping.items():
                if new_name in self.spot._joint_name_to_id:
                    joint_idx = self.spot._joint_name_to_id[new_name]
                    self.joints[old_name] = joint_idx
                    self._old_to_new_mapping[old_name] = new_name
                    self._new_to_old_mapping[new_name] = old_name
        
        # Also build direct mapping for new names
        if hasattr(self.spot, '_joint_name_to_id') and self.spot._joint_name_to_id:
            for new_name, joint_idx in self.spot._joint_name_to_id.items():
                if 'motor' in new_name.lower():
                    self.joints[new_name] = joint_idx
        
        print(f"[Adapter] Built joint mapping: {len(self.joints)} joints mapped")
        if len(self.joints) == 0:
            print("[Adapter] WARNING: No joints mapped! Joint control may not work.")
            print(f"[Adapter] Spot has _joint_name_to_id: {hasattr(self.spot, '_joint_name_to_id')}")
            if hasattr(self.spot, '_joint_name_to_id'):
                print(f"[Adapter] Available joint names: {list(self.spot._joint_name_to_id.keys())[:10]}...")
    
    def step(self):
        """Advance physics one step and update overlays/locomotion."""
        # Safety check
        if self.safety_enabled:
            self._check_safety()
        
        # Handle dynamic overlay
        self._handle_dynamic_overlay()
        
        # Handle locomotion if active
        if self._locomotion_active:
            self._update_locomotion()
        
        # Step the underlying environment
        # For bezier env, we need to pass joint angles as action array
        if hasattr(self.env, 'pass_joint_angles'):
            # Convert joint targets to action array
            action = self._joint_targets_to_action_array()
            # Pass joint angles to bezier env (it uses self.ja internally)
            self.env.pass_joint_angles(action)
            # Step with dummy action (bezier env uses self.ja from pass_joint_angles)
            self.env.step(np.zeros(12))
        else:
            # Regular gym env - use action array directly
            action = self._joint_targets_to_action_array()
            self.env.step(action)
    
    def _joint_targets_to_action_array(self) -> np.ndarray:
        """Convert joint targets dict to 12-motor action array."""
        # Get motor order from Spot
        # Import MOTOR_NAMES from spot module if available
        try:
            from spotmicro.spot import MOTOR_NAMES
            motor_names = MOTOR_NAMES
        except:
            # Fallback: use order from spot.py
            motor_names = [
                "motor_front_left_hip", "motor_front_left_upper_leg",
                "motor_front_left_lower_leg", "motor_front_right_hip",
                "motor_front_right_upper_leg", "motor_front_right_lower_leg",
                "motor_back_left_hip", "motor_back_left_upper_leg",
                "motor_back_left_lower_leg", "motor_back_right_hip",
                "motor_back_right_upper_leg", "motor_back_right_lower_leg"
            ]
        
        # Start with current joint angles as baseline
        if hasattr(self.spot, 'GetMotorAngles'):
            action = np.array(self.spot.GetMotorAngles())
            if len(action) != 12:
                action = np.zeros(12)
        else:
            action = np.zeros(12)
        
        # Fill action array from joint targets (overwrite with our targets)
        targets_applied = 0
        for i, motor_name in enumerate(motor_names):
            # Try to find matching old name first
            found = False
            for old_name, new_name in self._old_to_new_mapping.items():
                if new_name == motor_name and old_name in self._joint_targets:
                    action[i] = self._joint_targets[old_name]
                    found = True
                    targets_applied += 1
                    break
            
            # Or use new name directly
            if not found and motor_name in self._joint_targets:
                action[i] = self._joint_targets[motor_name]
                targets_applied += 1
        
        # Debug: warn if no targets were applied
        if len(self._joint_targets) > 0 and targets_applied == 0:
            print(f"[Adapter] WARNING: {len(self._joint_targets)} joint targets set but none applied to action array!")
            print(f"[Adapter] Joint targets: {list(self._joint_targets.keys())}")
            print(f"[Adapter] Mapping: {list(self._old_to_new_mapping.items())[:3]}")
        
        return action
    
    def _apply_joint_targets(self):
        """Apply current joint targets directly via PyBullet."""
        # For bezier env, we use action arrays instead of direct PyBullet calls
        # This method is kept for compatibility but the actual control happens via action array
        pass
    
    def set_joint_position(self, joint_name: str, target_pos: float):
        """Set a joint target position."""
        self._joint_targets[joint_name] = target_pos
        self.baseline_joint_targets[joint_name] = target_pos
        # Note: Actual application happens via action array in step() method
    
    def get_base_pose(self) -> Tuple[list, list]:
        """Get current base position and orientation."""
        return self.spot.GetBasePosition(), self.spot.GetBaseOrientation()
    
    def move_base_smooth(self, dy: float, duration_s: float = 1.5, steps: int = 120):
        """
        Smoothly move base position using BezierStepper for locomotion.
        
        Args:
            dy: Distance to move in Y direction (positive = forward, negative = backward)
            duration_s: Duration of movement in seconds
            steps: Number of simulation steps (ignored, uses BezierStepper)
        """
        # For significant movements, use BezierStepper
        if abs(dy) > 0.05:  # Threshold: use locomotion for movements > 5cm
            self._start_locomotion(dy, duration_s)
        else:
            # Small movements: use direct position manipulation
            self._move_base_direct(dy, duration_s)
    
    def _start_locomotion(self, dy: float, duration_s: float):
        """Start BezierStepper-based locomotion."""
        self._use_locomotion = True
        self._locomotion_target_dy = dy
        self._locomotion_duration = duration_s
        self._locomotion_start_time = time.time()
        self._locomotion_active = True
        
        # Convert dy to StepLength (BezierStepper parameter)
        # BezierStepper StepLength range: [-0.05, 0.05] meters per step
        # Scale dy to this range (dy is total distance, need per-step)
        steps_per_second = 1.0 / self.bezier_stepper.dt  # ~100 Hz
        total_steps = duration_s * steps_per_second
        step_length = dy / total_steps if total_steps > 0 else 0.0
        
        # Clamp to BezierStepper limits
        step_length = np.clip(step_length, -0.05, 0.05)
        
        # Set BezierStepper parameters
        self.bezier_stepper.StepLength = step_length
        self.bezier_stepper.StepVelocity = 0.3  # Moderate speed
        
        print(f"[Adapter] Starting locomotion: dy={dy:.3f}m, duration={duration_s:.2f}s, step_length={step_length:.4f}")
    
    def _update_locomotion(self):
        """Update BezierStepper locomotion each step."""
        elapsed = time.time() - self._locomotion_start_time
        
        if elapsed >= self._locomotion_duration:
            # Locomotion complete
            self._locomotion_active = False
            self._use_locomotion = False
            self.bezier_stepper.StepLength = 0.0
            return
        
        # Get BezierStepper parameters
        pos, orn, step_length, lateral_fraction, yaw_rate, step_velocity, clearance_height, penetration_depth = \
            self.bezier_stepper.return_bezier_params()
        
        # Update BezierStepper state machine
        # The environment's IK system will compute joint angles from BezierStepper parameters
        self.bezier_stepper.StateMachine()
    
    def _move_base_direct(self, dy: float, duration_s: float):
        """Move base using direct position manipulation (for small movements)."""
        base_pos, base_orn = self.get_base_pose()
        start_y = base_pos[1]
        target_y = start_y + dy
        
        # Interpolate over duration
        steps = int(duration_s * 240.0)  # 240 Hz physics
        for i in range(steps):
            t = i / steps if steps > 0 else 0.0
            smooth_t = t * t * (3.0 - 2.0 * t)  # Ease-in-out
            current_y = start_y + (target_y - start_y) * smooth_t
            new_pos = [base_pos[0], current_y, base_pos[2]]
            p.resetBasePositionAndOrientation(self.quadruped, new_pos, base_orn)
            # Step physics
            p.stepSimulation()
            if hasattr(self.env, 'gui') and self.env.gui:
                time.sleep(1.0 / 240.0)
    
    def hold(self, duration_s: float, steps_per_second: int = 60):
        """Hold current pose for specified duration."""
        total_steps = int(duration_s * steps_per_second)
        dt = 1.0 / steps_per_second
        
        for _ in range(total_steps):
            self.step()
            if hasattr(self.env, 'gui') and self.env.gui:
                time.sleep(dt)
    
    def apply_yaw_scan(self, yaw_angle_rad: float):
        """Apply yaw rotation to base for scanning motion."""
        base_pos, base_orn = self.get_base_pose()
        euler = p.getEulerFromQuaternion(base_orn)
        new_euler = [euler[0], euler[1], euler[2] + yaw_angle_rad]
        new_orn = p.getQuaternionFromEuler(new_euler)
        p.resetBasePositionAndOrientation(self.quadruped, base_pos, new_orn)
    
    def stand(self):
        """Set joints to standard standing configuration."""
        stand_config = {
            'fl.hx': 0.0, 'fl.hy': 0.6, 'fl.kn': -1.4,
            'fr.hx': 0.0, 'fr.hy': 0.6, 'fr.kn': -1.4,
            'hl.hx': 0.0, 'hl.hy': 0.6, 'hl.kn': -1.4,
            'hr.hx': 0.0, 'hr.hy': 0.6, 'hr.kn': -1.4
        }
        self._apply_config(stand_config)
    
    def sit(self):
        """Set joints to sitting configuration - EXAGGERATED for visibility."""
        sit_config = {
            'fl.hx': 0.0, 'fl.hy': 1.2, 'fl.kn': -2.8,  # More pronounced sit
            'fr.hx': 0.0, 'fr.hy': 1.2, 'fr.kn': -2.8,
            'hl.hx': 0.0, 'hl.hy': 1.3, 'hl.kn': -3.0,
            'hr.hx': 0.0, 'hr.hy': 1.3, 'hr.kn': -3.0
        }
        self._apply_config(sit_config)
    
    def emotional_pose(self, valence: float, arousal: float):
        """Map valence/arousal to body language."""
        # Threshold logic for strong emotions
        if valence > 0.7 and arousal > 0.7:
            config = {
                'fl.hy': 0.4, 'fl.kn': -1.0,
                'fr.hy': 0.4, 'fr.kn': -1.0,
                'hl.hy': 0.8, 'hl.kn': -1.8,
                'hr.hy': 0.8, 'hr.kn': -1.8
            }
            self._apply_config(config)
            return
        
        if valence < -0.7 and arousal < -0.7:
            config = {
                'fl.hy': 1.0, 'fl.kn': -2.2,
                'fr.hy': 1.0, 'fr.kn': -2.2,
                'hl.hy': 1.1, 'hl.kn': -2.4,
                'hr.hy': 1.1, 'hr.kn': -2.4
            }
            self._apply_config(config)
            return
        
        # Continuous mapping
        base_hy = 0.6
        base_kn = -1.4
        
        valence = np.clip(valence, -0.7, 0.7)
        arousal = np.clip(arousal, -0.7, 0.7)
        
        pitch_bias = valence * 0.4
        height_bias = arousal * 0.2
        
        front_mod = -pitch_bias + height_bias
        hind_mod = pitch_bias + height_bias
        
        config = {
            'fl.hy': base_hy - front_mod, 'fl.kn': base_kn - front_mod,
            'fr.hy': base_hy - front_mod, 'fr.kn': base_kn - front_mod,
            'hl.hy': base_hy - hind_mod, 'hl.kn': base_kn - hind_mod,
            'hr.hy': base_hy - hind_mod, 'hr.kn': base_kn - hind_mod
        }
        self._apply_config(config)
    
    def _apply_config(self, config: Dict[str, float]):
        """Apply joint configuration."""
        for name, pos in config.items():
            self.set_joint_position(name, pos)
    
    def set_baseline_pose(self, intent_or_name: str):
        """Set baseline pose by intent/name."""
        name = intent_or_name.upper()
        if name not in self.baseline_poses:
            # Fallbacks
            if "DE_ESCALATE" in name or "DE-ESCALATE" in name:
                name = "DE_ESCALATE"
            elif "CHECK_IN" in name or "CHECK-IN" in name or "SUPPORT" in name:
                name = "CHECK_IN"
            elif "ENGAGE" in name or "PLAYFUL" in name:
                name = "ENGAGE"
            elif "CAUTION" in name:
                name = "CAUTION"
            else:
                name = "NEUTRAL"
        
        self.baseline_pose_name = name
        config = self.baseline_poses.get(name, self.baseline_poses["NEUTRAL"])
        self._apply_config(config)
        
        # Refresh cache
        self.baseline_joint_targets = {}
        for logical_name, angle in config.items():
            self.baseline_joint_targets[logical_name] = angle
        
        print(f"[Adapter] Baseline pose set to {self.baseline_pose_name}")
    
    def start_overlay(self, name: str, duration: float, params: Dict):
        """Start a dynamic overlay animation."""
        self.active_overlay = name.upper()
        self.overlay_start_time = time.time()
        self.overlay_duration = max(0.1, float(duration))
        self.overlay_params = params or {}
        print(f"[Adapter] Starting overlay {self.active_overlay} for {self.overlay_duration:.2f}s")
    
    def stop_overlay(self):
        """Stop any active overlay."""
        if self.active_overlay is not None:
            print(f"[Adapter] Stopping overlay {self.active_overlay}")
        self.active_overlay = None
        self.overlay_params = {}
        self.set_baseline_pose(self.baseline_pose_name)
    
    def _handle_dynamic_overlay(self):
        """Apply time-varying joint deltas each physics step."""
        if not self.active_overlay:
            return
        
        elapsed = time.time() - self.overlay_start_time
        if elapsed > self.overlay_duration:
            self.stop_overlay()
            return
        
        freq = float(self.overlay_params.get("freq", 2.0))
        amp = float(self.overlay_params.get("amplitude", 0.1))
        phase = 2.0 * np.pi * freq * elapsed
        sin_val = np.sin(phase)
        
        name = self.active_overlay
        
        # Dispatch to intent-specific overlay implementations
        if name == "DE_ESCALATE":
            self._overlay_de_escalate(amp, sin_val, elapsed)
        elif name == "CAUTION":
            self._overlay_caution(amp, sin_val, elapsed)
        elif name == "CHECK_IN":
            self._overlay_check_in(amp, sin_val, elapsed)
        elif name == "ENGAGE":
            self._overlay_engage(amp, sin_val, elapsed)
        elif name == "SHIVER":
            self._overlay_shiver(amp, sin_val, elapsed)
    
    def _get_baseline_target(self, joint_key: str, default: float = 0.0) -> float:
        """Get baseline target for a joint."""
        return self.baseline_joint_targets.get(joint_key, default)
    
    def _apply_delta_config(self, deltas: Dict[str, float]):
        """Apply small deltas on top of cached baseline targets."""
        intensity = getattr(self, "motion_intensity", 1.0)
        if intensity <= 0.0:
            return
        
        for joint_key, delta in deltas.items():
            scaled_delta = delta * intensity
            base = self.baseline_joint_targets.get(joint_key, 0.0)
            self.set_joint_position(joint_key, base + scaled_delta)
    
    def _overlay_de_escalate(self, amp: float, sin_val: float, elapsed: float):
        """DE_ESCALATE overlay: slow settling."""
        settle_factor = np.exp(-elapsed * 1.5)
        slow_component = amp * 0.5 * (1.0 - settle_factor)
        osc_component = amp * 0.3 * settle_factor * sin_val
        total = -abs(slow_component) + osc_component
        
        deltas = {
            'fl.hy': total, 'fr.hy': total,
            'hl.hy': total * 1.1, 'hr.hy': total * 1.1,
            'fl.kn': -total * 1.2, 'fr.kn': -total * 1.2,
            'hl.kn': -total * 1.3, 'hr.kn': -total * 1.3,
        }
        self._apply_delta_config(deltas)
    
    def _overlay_caution(self, amp: float, sin_val: float, elapsed: float):
        """CAUTION overlay: scan."""
        left = amp * sin_val
        right = -amp * sin_val
        deltas = {
            'fl.hy': left, 'fr.hy': right,
            'fl.kn': -left * 0.5, 'fr.kn': -right * 0.5,
        }
        self._apply_delta_config(deltas)
        
        # Yaw oscillation
        yaw_amp = self.overlay_params.get("yaw_amplitude", 0.1)
        yaw = yaw_amp * sin_val
        base_pos, base_orn = self.get_base_pose()
        euler = p.getEulerFromQuaternion(base_orn)
        new_orn = p.getQuaternionFromEuler([euler[0], euler[1], euler[2] + yaw])
        p.resetBasePositionAndOrientation(self.quadruped, base_pos, new_orn)
    
    def _overlay_check_in(self, amp: float, sin_val: float, elapsed: float):
        """CHECK_IN overlay: gentle breathe."""
        breathe = amp * sin_val
        deltas = {
            'fl.hy': -breathe * 0.6, 'fr.hy': -breathe * 0.6,
            'fl.kn': breathe * 0.8, 'fr.kn': breathe * 0.8,
        }
        self._apply_delta_config(deltas)
    
    def _overlay_engage(self, amp: float, sin_val: float, elapsed: float):
        """ENGAGE overlay: energetic bounce."""
        bounce = amp * sin_val
        deltas = {
            'fl.hy': -bounce, 'fr.hy': -bounce,
            'hl.hy': -bounce * 0.8, 'hr.hy': -bounce * 0.8,
            'fl.kn': bounce * 1.2, 'fr.kn': bounce * 1.2,
            'hl.kn': bounce * 1.0, 'hr.kn': bounce * 1.0,
        }
        self._apply_delta_config(deltas)
    
    def _overlay_shiver(self, amp: float, sin_val: float, elapsed: float):
        """High-frequency tremor."""
        tremor = amp * 0.3 * sin_val
        deltas = {
            'fl.hy': tremor, 'fr.hy': -tremor,
            'hl.hy': -tremor, 'hr.hy': tremor,
            'fl.kn': -tremor, 'fr.kn': tremor,
            'hl.kn': tremor, 'hr.kn': -tremor,
        }
        self._apply_delta_config(deltas)
    
    def apply_reaction_pose(self, reaction_type: str):
        """Apply a pose based on reaction type - EXAGGERATED for visibility."""
        reaction_lower = reaction_type.lower()
        
        if 'de-escalate' in reaction_lower:
            # Clear crouch pose
            config = {
                'fl.hx': 0.0, 'fl.hy': 1.0, 'fl.kn': -2.2,
                'fr.hx': 0.0, 'fr.hy': 1.0, 'fr.kn': -2.2,
                'hl.hx': 0.0, 'hl.hy': 1.1, 'hl.kn': -2.4,
                'hr.hx': 0.0, 'hr.hy': 1.1, 'hr.kn': -2.4
            }
            self._apply_config(config)
        elif 'check-in' in reaction_lower or 'supportive' in reaction_lower:
            # Forward lean
            config = {
                'fl.hx': 0.0, 'fl.hy': 0.8, 'fl.kn': -1.8,
                'fr.hx': 0.0, 'fr.hy': 0.8, 'fr.kn': -1.8,
                'hl.hx': 0.0, 'hl.hy': 0.4, 'hl.kn': -1.0,
                'hr.hx': 0.0, 'hr.hy': 0.4, 'hr.kn': -1.0
            }
            self._apply_config(config)
        elif 'engage' in reaction_lower or 'playful' in reaction_lower:
            # Tall and open
            config = {
                'fl.hx': 0.0, 'fl.hy': 0.3, 'fl.kn': -0.8,
                'fr.hx': 0.0, 'fr.hy': 0.3, 'fr.kn': -0.8,
                'hl.hx': 0.0, 'hl.hy': 0.8, 'hl.kn': -1.8,
                'hr.hx': 0.0, 'hr.hy': 0.8, 'hr.kn': -1.8
            }
            self._apply_config(config)
        elif 'caution' in reaction_lower:
            # Alert stance
            config = {
                'fl.hx': 0.0, 'fl.hy': 0.4, 'fl.kn': -1.0,
                'fr.hx': 0.0, 'fr.hy': 0.4, 'fr.kn': -1.0,
                'hl.hx': 0.0, 'hl.hy': 0.7, 'hl.kn': -1.6,
                'hr.hx': 0.0, 'hr.hy': 0.7, 'hr.kn': -1.6
            }
            self._apply_config(config)
        else:
            self.stand()
    
    def stop(self):
        """Stop all movement."""
        self.is_moving = False
        self.current_speed = 0.0
        self.sit()
    
    def set_speed(self, speed_multiplier: float):
        """Set movement speed multiplier."""
        self.current_speed = np.clip(speed_multiplier, 0.0, 1.0)
    
    def adjust_distance(self, distance_multiplier: float):
        """Adjust distance from target."""
        self.current_distance = np.clip(distance_multiplier, 0.5, 2.0)
    
    def move_backward(self, distance: float = 0.2):
        """Move backward."""
        self.move_base_smooth(-distance, duration_s=1.0)
    
    def move_forward(self, distance: float = 0.2):
        """Move forward."""
        self.move_base_smooth(distance, duration_s=1.0)
    
    def pause(self):
        """Pause movement."""
        self.is_moving = False
        self.current_speed = 0.0
    
    def continue_movement(self):
        """Resume movement."""
        self.is_moving = True
        self.current_speed = 1.0
        self.stand()
    
    def set_motion_intensity(self, intensity: float):
        """Set global motion intensity scalar."""
        self.motion_intensity = np.clip(intensity, 0.0, 1.0)
    
    def clamp_overlay_params(self, params: Dict) -> Dict:
        """Clamp overlay parameters to safe range."""
        clamped = params.copy()
        if "amplitude" in clamped:
            clamped["amplitude"] = min(clamped["amplitude"], self.safe_overlay_amplitude)
        if "freq" in clamped:
            clamped["freq"] = np.clip(clamped["freq"], 
                                      self.safe_overlay_freq_range[0],
                                      self.safe_overlay_freq_range[1])
        return clamped
    
    def _check_safety(self):
        """Safety monitoring (warnings only)."""
        if not self.safety_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_safety_check_time < self.safety_check_interval:
            return
        
        self.last_safety_check_time = current_time
        
        # Check base orientation
        base_pos, base_orn = self.get_base_pose()
        euler = p.getEulerFromQuaternion(base_orn)
        roll, pitch, yaw = euler
        
        if abs(roll) > self.max_roll_pitch or abs(pitch) > self.max_roll_pitch:
            pass  # Warning only
        
        # Check overlay amplitude
        if self.active_overlay:
            amp = self.overlay_params.get("amplitude", 0.0)
            if amp > self.safe_overlay_amplitude:
                self.overlay_params["amplitude"] = min(amp, self.safe_overlay_amplitude)
