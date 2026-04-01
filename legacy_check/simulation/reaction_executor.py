"""
Reaction Executor Module

State machine + trajectory generator that safely executes ReactionAction objects
with smooth blending, rate limiting, and safety monitoring.
"""

import time
import numpy as np
import pybullet as p
from enum import Enum
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Use relative imports
from ..core.reaction_action import ReactionAction
from .behavior_renderer import BehaviorRenderer


class ReactionState(Enum):
    """State machine states for reaction execution."""
    IDLE = "idle"
    EXECUTING = "executing"
    BLENDING_OUT = "blending_out"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    time: float
    joint_targets: Dict[str, float]
    base_target: Optional[Tuple[float, float, float]] = None
    overlay_params: Optional[Dict] = None


class ReactionExecutor:
    """
    Executes ReactionAction objects with:
    - State machine (IDLE, EXECUTING, BLENDING_OUT)
    - Minimum dwell times (prevent rapid switching)
    - Smooth trajectory generation (blend between reactions)
    - Rate limiting on motion parameters
    - Safety monitoring
    """
    
    def __init__(self, 
                 env,
                 renderer: BehaviorRenderer,
                 min_dwell_time: float = 1.5,
                 blend_duration: float = 0.8,
                 max_joint_velocity: float = 2.0,  # rad/s
                 max_base_velocity: float = 0.3,   # m/s
                 verbose: bool = True):
        """
        Initialize reaction executor.
        
        Args:
            env: SpotEnv-compatible instance (NewSimSpotAdapter or old SpotEnv)
            renderer: BehaviorRenderer instance
            min_dwell_time: Minimum time to execute a reaction before allowing new one (seconds)
            blend_duration: Duration for blending between reactions (seconds)
            max_joint_velocity: Maximum joint angular velocity (rad/s)
            max_base_velocity: Maximum base translation velocity (m/s)
            verbose: If True, print debug output
        """
        self.env = env
        self.renderer = renderer
        self.min_dwell_time = min_dwell_time
        self.blend_duration = blend_duration * 2.5  # Maximum blend duration (1.2s default -> 3.0s) for ultra-smooth transitions
        self.max_joint_velocity = max_joint_velocity * 0.4  # Further reduced to 40% (was 50%, now 40% total) for ultra-smooth motion
        self.max_base_velocity = max_base_velocity * 0.4  # Further reduced to 40% (was 50%, now 40% total) for ultra-smooth motion
        self.verbose = verbose
        
        # State machine
        self.state = ReactionState.IDLE
        self.current_action: Optional[ReactionAction] = None
        self.current_intent: Optional[str] = None
        self.state_start_time: float = 0.0
        
        # Trajectory tracking
        self.trajectory: list = []
        self.trajectory_start_time: float = 0.0
        self.current_baseline_pose: Dict[str, float] = {}
        self.target_baseline_pose: Dict[str, float] = {}
        
        # Rate limiting state
        self.last_speed_mult: float = 0.5
        self.last_distance_mult: float = 1.0
        self.last_overlay_amplitude: float = 0.0
    
    def can_start_reaction(self, action: ReactionAction) -> Tuple[bool, str]:
        """
        Check if a new reaction can start (respects dwell time, priority).
        
        Args:
            action: Proposed ReactionAction
        
        Returns:
            Tuple of (can_start: bool, reason: str)
        """
        # Emergency de-escalate can always pre-empt
        if action.intent == "DE_ESCALATE" and self.state != ReactionState.EMERGENCY_STOP:
            return (True, "Emergency de-escalate pre-empts")
        
        # Check state
        if self.state == ReactionState.EMERGENCY_STOP:
            return (False, "Emergency stop active")
        
        if self.state == ReactionState.IDLE:
            return (True, "Idle state")
        
        # Check dwell time
        elapsed = time.time() - self.state_start_time
        if elapsed < self.min_dwell_time:
            return (False, f"Dwell time not met ({elapsed:.2f}s < {self.min_dwell_time:.2f}s)")
        
        # Same intent: allow if dwell time met
        if self.current_intent == action.intent:
            return (True, "Same intent, dwell time met")
        
        # Different intent: allow transition
        return (True, "Different intent, transitioning")
    
    def start_reaction(self, action: ReactionAction, 
                      valence: Optional[float] = None,
                      arousal: Optional[float] = None) -> bool:
        """
        Start executing a new reaction (with blending if needed).
        
        Args:
            action: ReactionAction to execute
            valence: Optional valence for VA_POSE mode
            arousal: Optional arousal for VA_POSE mode
        
        Returns:
            True if reaction started, False if rejected
        """
        can_start, reason = self.can_start_reaction(action)
        if not can_start:
            if self.verbose:
                print(f"[EXECUTOR] Rejected reaction {action.intent}: {reason}")
            return False
        
        if self.verbose:
            print(f"[EXECUTOR] Starting reaction: {action.intent} ({reason})")
        
        # If currently executing, blend out first
        if self.state == ReactionState.EXECUTING:
            self._blend_to_new_reaction(action, valence, arousal)
        else:
            # Direct start
            self._start_direct(action, valence, arousal)
        
        return True
    
    def _start_direct(self, action: ReactionAction, 
                     valence: Optional[float], 
                     arousal: Optional[float]):
        """Start reaction directly (no blending needed)."""
        self.current_action = action
        self.current_intent = action.intent
        self.state = ReactionState.EXECUTING
        self.state_start_time = time.time()
        
        # Apply rate-limited parameters
        speed_mult = self._rate_limit_speed(action.speed_mult)
        distance_mult = self._rate_limit_distance(action.distance_mult)
        
        # Create rate-limited action copy
        rate_limited_action = ReactionAction(
            intent=action.intent,
            speed_mult=speed_mult,
            distance_mult=distance_mult,
            pose_mode=action.pose_mode,
            pose_name=action.pose_name,
            duration_s=action.duration_s,
            explain=action.explain + " [rate-limited]",
            debug=action.debug
        )
        
        # Render with trajectory generation
        self._render_with_trajectory(rate_limited_action, valence, arousal)
    
    def _blend_to_new_reaction(self, new_action: ReactionAction,
                               valence: Optional[float],
                               arousal: Optional[float]):
        """Blend from current reaction to new one."""
        if self.verbose:
            print(f"[EXECUTOR] Blending from {self.current_intent} to {new_action.intent}")
        
        # Generate blend trajectory
        self._generate_blend_trajectory(new_action, valence, arousal)
        
        # Update state
        self.current_action = new_action
        self.current_intent = new_action.intent
        self.state = ReactionState.BLENDING_OUT
        self.state_start_time = time.time()
    
    def _render_with_trajectory(self, action: ReactionAction,
                               valence: Optional[float],
                               arousal: Optional[float]):
        """Render action and generate smooth trajectory."""
        # Get current baseline pose
        self.current_baseline_pose = self.env.baseline_joint_targets.copy()
        
        # Get target baseline pose
        target_pose_name = action.pose_name if action.pose_mode == "REACTION_POSE" else action.intent
        if target_pose_name in self.env.baseline_poses:
            self.target_baseline_pose = self.env.baseline_poses[target_pose_name].copy()
        else:
            # Fallback: use current
            self.target_baseline_pose = self.current_baseline_pose.copy()
        
        # Generate trajectory for baseline pose transition
        self._generate_baseline_trajectory(action.duration_s)
        
        # Render action (this sets baseline and overlay)
        self.renderer.render(action, valence=valence, arousal=arousal)
        
        # Store trajectory start time
        self.trajectory_start_time = time.time()
    
    def _generate_baseline_trajectory(self, duration_s: float):
        """Generate smooth trajectory for baseline pose transition."""
        # Use minimum-jerk interpolation (smooth acceleration)
        num_points = max(10, int(duration_s * 20))  # 20 Hz trajectory
        
        self.trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.0
            
            # Minimum-jerk: t^3 * (6t^2 - 15t + 10)
            smooth_t = t * t * t * (6.0 * t * t - 15.0 * t + 10.0)
            
            # Interpolate joint targets
            joint_targets = {}
            for joint_name in self.target_baseline_pose:
                if joint_name in self.current_baseline_pose:
                    start_val = self.current_baseline_pose[joint_name]
                    end_val = self.target_baseline_pose[joint_name]
                    joint_targets[joint_name] = start_val + (end_val - start_val) * smooth_t
                else:
                    joint_targets[joint_name] = self.target_baseline_pose[joint_name]
            
            traj_point = TrajectoryPoint(
                time=t * duration_s,
                joint_targets=joint_targets
            )
            self.trajectory.append(traj_point)
    
    def _generate_blend_trajectory(self, new_action: ReactionAction,
                                   valence: Optional[float],
                                   arousal: Optional[float]):
        """Generate blend trajectory from current to new reaction."""
        # Get current and target poses
        current_pose = self.env.baseline_joint_targets.copy()
        target_pose_name = new_action.pose_name if new_action.pose_mode == "REACTION_POSE" else new_action.intent
        if target_pose_name in self.env.baseline_poses:
            target_pose = self.env.baseline_poses[target_pose_name].copy()
        else:
            target_pose = current_pose.copy()
        
        # Generate blend trajectory
        num_points = max(10, int(self.blend_duration * 20))
        self.trajectory = []
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.0
            smooth_t = t * t * t * (6.0 * t * t - 15.0 * t + 10.0)  # Minimum-jerk
            
            joint_targets = {}
            for joint_name in target_pose:
                if joint_name in current_pose:
                    start_val = current_pose[joint_name]
                    end_val = target_pose[joint_name]
                    joint_targets[joint_name] = start_val + (end_val - start_val) * smooth_t
                else:
                    joint_targets[joint_name] = target_pose[joint_name]
            
            traj_point = TrajectoryPoint(
                time=t * self.blend_duration,
                joint_targets=joint_targets
            )
            self.trajectory.append(traj_point)
        
        # After blend, render new action
        self.renderer.render(new_action, valence=valence, arousal=arousal)
        self.trajectory_start_time = time.time()
    
    def update(self, dt: float):
        """
        Update executor (call every control loop iteration).
        
        Args:
            dt: Time step (seconds)
        """
        if self.state == ReactionState.IDLE:
            return
        
        elapsed = time.time() - self.state_start_time
        
        # Check if reaction finished
        if self.current_action:
            if elapsed >= self.current_action.duration_s:
                if self.state == ReactionState.BLENDING_OUT:
                    # Blend complete, transition to executing
                    self.state = ReactionState.EXECUTING
                    self.state_start_time = time.time()
                else:
                    # Reaction complete, return to idle
                    self._finish_reaction()
                    return
        
        # Apply trajectory if blending
        if self.state == ReactionState.BLENDING_OUT and self.trajectory:
            self._apply_trajectory()
    
    def _apply_trajectory(self):
        """Apply current trajectory point to robot."""
        if not self.trajectory:
            return
        
        elapsed = time.time() - self.trajectory_start_time
        
        # Find trajectory point
        traj_idx = min(int(elapsed * 20), len(self.trajectory) - 1)  # 20 Hz
        if traj_idx < len(self.trajectory):
            point = self.trajectory[traj_idx]
            
            # Apply joint targets with velocity limiting
            for joint_name, target in point.joint_targets.items():
                self._set_joint_safe(joint_name, target)
    
    def _set_joint_safe(self, joint_name: str, target: float):
        """Set joint target with velocity limiting."""
        # Get current joint state
        if joint_name in self.env.joints:
            joint_idx = self.env.joints[joint_name]
            try:
                current_state = p.getJointState(self.env.spotId, joint_idx)
                current_pos = current_state[0]
                
                # Compute desired velocity (assuming 240 Hz physics)
                dt = 1.0 / 240.0
                desired_velocity = (target - current_pos) / dt
                
                # Clamp velocity
                if abs(desired_velocity) > self.max_joint_velocity:
                    # Scale target to respect velocity limit
                    max_delta = self.max_joint_velocity * dt
                    target = current_pos + np.sign(target - current_pos) * max_delta
            except:
                # Fallback if joint state unavailable
                pass
            
            # Set joint
            self.env.set_joint_position(joint_name, target)
    
    def _rate_limit_speed(self, new_speed: float) -> float:
        """Rate limit speed multiplier changes."""
        max_delta = 0.3  # Max change per update
        delta = new_speed - self.last_speed_mult
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
        self.last_speed_mult += delta
        return np.clip(self.last_speed_mult, 0.0, 1.0)
    
    def _rate_limit_distance(self, new_distance: float) -> float:
        """Rate limit distance multiplier changes."""
        max_delta = 0.2
        delta = new_distance - self.last_distance_mult
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
        self.last_distance_mult += delta
        return np.clip(self.last_distance_mult, 0.5, 2.0)
    
    def _finish_reaction(self):
        """Finish current reaction and return to idle."""
        if self.verbose:
            print(f"[EXECUTOR] Reaction {self.current_intent} finished, returning to idle")
        
        self.state = ReactionState.IDLE
        self.current_action = None
        self.current_intent = None
        self.trajectory = []
    
    def emergency_stop(self):
        """Trigger emergency stop (e.g., from safety monitor)."""
        if self.verbose:
            print("[EXECUTOR] EMERGENCY STOP triggered")
        
        self.state = ReactionState.EMERGENCY_STOP
        self.env.stop()  # Use env's stop method
    
    def get_state(self) -> Dict:
        """Get current executor state (for debugging)."""
        return {
            'state': self.state.value,
            'current_intent': self.current_intent,
            'elapsed_time': time.time() - self.state_start_time if self.state_start_time > 0 else 0.0,
            'has_trajectory': len(self.trajectory) > 0
        }
