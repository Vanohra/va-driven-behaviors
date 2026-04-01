"""
Behavior Renderer Module

Renders robot behaviors in simulation according to the Reaction-to-Motion Contract.
Each reaction produces a visibly distinct, time-extended physical behavior.
"""

import time
import numpy as np
import pybullet as p
from typing import Optional, Dict

# Use relative imports for core modules
from ..core.reaction_action import ReactionAction


class BehaviorRenderer:
    """
    Renders structured actions as visible behaviors in simulation.
    
    Implements the Reaction-to-Motion Contract:
    - Each reaction unfolds over time (≥1 second)
    - Involves at least two physical variables
    - Is visually distinguishable within 1-2 seconds
    """
    
    def __init__(self, env, verbose: bool = True):
        """
        Initialize behavior renderer.
        
        Args:
            env: SpotEnv-compatible instance (NewSimSpotAdapter or old SpotEnv)
            verbose: If True, print debug output
        """
        self.env = env
        self.verbose = verbose
        self.animation_start_time = None
        self.current_action: Optional[ReactionAction] = None
        self.initial_base_y = None
        self.initial_body_height = None
    
    def render(self, action: ReactionAction, valence: Optional[float] = None, arousal: Optional[float] = None):
        """
        Render an action with pose, animation, and movement.
        
        Args:
            action: ReactionAction to render
            valence: Optional valence for VA_POSE mode
            arousal: Optional arousal for VA_POSE mode
        """
        self.current_action = action
        self.animation_start_time = time.time()
        
        # Save initial state
        base_pos, _ = self.env.get_base_pose()
        self.initial_base_y = base_pos[1]
        
        if self.verbose:
            print(f"[RENDER] {action.intent} | pose={action.pose_mode} | "
                  f"speed={action.speed_mult:.2f} | distance={action.distance_mult:.2f} | "
                  f"duration={action.duration_s:.2f}s | {action.explain}")
        
        # A) Set baseline pose first
        self._apply_baseline_pose(action, valence, arousal)

        # B) Trigger dynamic overlay on top of baseline
        self._trigger_overlay(action, valence, arousal)

        # C) Apply visible base motion (retreat/approach) as a separate smooth movement
        self._apply_base_motion(action)

    def _apply_baseline_pose(self, action: ReactionAction, valence: Optional[float], arousal: Optional[float]):
        """
        Set a stable baseline posture in the environment.

        - REACTION_POSE: uses intent-based baseline pose.
        - VA_POSE: uses emotional_pose but still remembers a fallback baseline name.
        """
        if action.pose_mode == "REACTION_POSE":
            # Use reaction pose based on intent name
            pose_name = action.pose_name or action.intent
            self.env.set_baseline_pose(pose_name)

        elif action.pose_mode == "VA_POSE":
            # Use continuous VA pose for posture, but remember that we're in VA baseline
            if valence is not None and arousal is not None:
                self.env.emotional_pose(valence, arousal)
                # Mark generic VA baseline so overlays can re-center later
                self.env.baseline_pose_name = "VA_POSE"
            else:
                pose_name = action.pose_name or action.intent
                self.env.set_baseline_pose(pose_name)

    def _overlay_params_for_action(self, action: ReactionAction, volatility: float = 0.0) -> Dict:
        """Derive overlay amplitude/frequency parameters from action and volatility."""
        # Map arousal/speed to frequency and amplitude - REDUCED for subtlety
        speed = action.speed_mult
        # Base ranges per intent - SLOWER frequencies for smoother motion
        intent = action.intent
        if intent == "DE_ESCALATE":
            base_freq = 0.5  # Slower for smoother motion (reduced from 0.7)
            base_amp = 0.03
        elif intent == "CAUTION":
            base_freq = 0.8  # Slower for smoother motion (reduced from 1.2)
            base_amp = 0.04
        elif intent == "CHECK_IN":
            base_freq = 0.5  # Slower for smoother motion (reduced from 0.7)
            base_amp = 0.025
        elif intent == "ENGAGE":
            base_freq = 0.9  # Slower for smoother motion (reduced from 1.2)
            base_amp = 0.05
        else:  # NEUTRAL
            base_freq = 0.3  # Very slow for ultra-smooth (reduced from 0.4)
            base_amp = 0.015

        # Modulate by speed/volatility - very conservative, minimal variation
        freq = base_freq * (0.9 + 0.2 * speed)  # Very tight range: 0.9-1.1x for smoothness
        amp = base_amp * (0.9 + 0.2 * speed)    # Very tight range: 0.9-1.1x for smoothness

        # Apply global motion intensity scalar from env (primary user-facing knob)
        motion_intensity = getattr(self.env, "motion_intensity", 1.0)
        amp *= motion_intensity

        params: Dict = {
            "freq": freq,
            "amplitude": amp,
        }

        # Extra yaw amplitude for CAUTION scan - further reduced
        if intent == "CAUTION":
            params["yaw_amplitude"] = 0.03  # Further reduced from 0.06 (50% more reduction)

        # Optional volatility-based SHIVER overlay may be triggered externally
        return params

    def _trigger_overlay(self, action: ReactionAction, valence: Optional[float], arousal: Optional[float]):
        """Start a named overlay corresponding to the action intent."""
        params = self._overlay_params_for_action(
            action,
            volatility=action.debug.get("volatility", 0.0) if action.debug else 0.0,
        )
        # Clamp overlay params to safe modulation subspace
        params = self.env.clamp_overlay_params(params)
        # Overlay name is just the intent (DE_ESCALATE, CAUTION, etc.)
        self.env.start_overlay(name=action.intent, duration=action.duration_s, params=params)

    def _apply_base_motion(self, action: ReactionAction):
        """
        Handle distance/motion visibly using base translation.

        REDUCED motion distances for subtlety:
        - DE_ESCALATE: retreat 0.15-0.3m (reduced from 0.3-0.6m).
        - ENGAGE: approach 0.08-0.15m (reduced from 0.15-0.3m).
        - Others: very small or no motion.
        """
        dy = 0.0
        if action.intent == "DE_ESCALATE":
            # Map distance_mult 1.5-2.0 to 0.08-0.15m backward (further reduced: 50% more)
            if action.distance_mult >= 1.5:
                t = min(1.0, (action.distance_mult - 1.5) / 0.5)
                dy = -(0.08 + 0.07 * t)  # Further reduced: 0.08-0.15m instead of 0.15-0.3m
        elif action.intent == "ENGAGE":
            # Approach if safe (distance_mult < 1.0) - further reduced distances
            if action.distance_mult < 1.0:
                t = min(1.0, (1.0 - action.distance_mult) / 0.3)  # 0.7-1.0
                dy = 0.04 + 0.04 * t  # Further reduced: 0.04-0.08m instead of 0.08-0.15m
        elif action.intent == "CAUTION":
            if 1.0 < action.distance_mult <= 1.5:
                t = (action.distance_mult - 1.0) / 0.5
                dy = -0.05 * t  # Further reduced: max 0.05m instead of 0.1m

        if abs(dy) > 1e-3:
            # Scale by global motion intensity so base motions follow same expressiveness knob
            motion_intensity = getattr(self.env, "motion_intensity", 1.0)
            dy *= motion_intensity

            # Use a slightly more pronounced motion for visual clarity
            duration = max(1.0, min(2.0, action.duration_s))
            # move_base_smooth itself steps simulation; caller must still drive high-rate physics for overlays
            self.env.move_base_smooth(dy=dy, duration_s=duration)
    
    def apply_animation(self, elapsed_time: float, duration_s: float):
        """
        Apply time-extended motion animation based on current intent.
        
        This implements the exact physical signatures from the contract.
        
        Args:
            elapsed_time: Time elapsed since animation started
            duration_s: Total duration of the action
        """
        # NOTE:
        # The fine-grained joint animation is now handled inside SpotEnv._handle_dynamic_overlay()
        # which is called every env.step(). BehaviorRenderer no longer drives per-step joint targets
        # directly; it only configures baseline + overlays + base motion.
        if self.current_action is None:
            return

        # Kept for backward-compat API; no-op under overlay-based engine.
        _ = self.current_action.intent
        _ = elapsed_time
        _ = duration_s
    
    def _render_de_escalate(self, progress: float, elapsed_time: float, duration_s: float):
        """
        DE_ESCALATE: Contract implementation
        
        Physical signature (all required):
        - Posture: body height decreases gradually (crouch)
        - Distance: smooth backward base motion (0.3-0.6 m)
        - Motion energy: decays to minimal
        - Temporal cue: settling motion at end (small downward ease)
        """
        # 1. Posture: Gradually lower body height (crouch)
        # Crouch factor: 0.0 (normal) -> 1.0 (fully crouched) over first 60% of duration
        crouch_progress = min(1.0, progress / 0.6)
        crouch_factor = crouch_progress  # Ease-in
        body_height_offset = -0.4 * crouch_factor  # Lower by 0.4 units
        
        # Apply crouch to all legs
        base_hy = 0.6 + body_height_offset
        base_kn = -1.4 - 0.6 * crouch_factor  # More bent knees
        crouch_config = {
            'fl.hy': base_hy, 'fl.kn': base_kn,
            'fr.hy': base_hy, 'fr.kn': base_kn,
            'hl.hy': base_hy + 0.1, 'hl.kn': base_kn - 0.2,
            'hr.hy': base_hy + 0.1, 'hr.kn': base_kn - 0.2
        }
        self.env._apply_config(crouch_config)
        
        # 2. Distance: Smooth backward motion (0.3-0.6 m)
        # Move backward over first 70% of duration
        # Map distance_mult: 1.5-2.0 -> 0.3-0.6m retreat
        if progress < 0.7 and self.current_action.distance_mult >= 1.5:
            move_progress = progress / 0.7
            # Ease-in-out for smooth motion
            ease_t = move_progress * move_progress * (3.0 - 2.0 * move_progress)
            # Map distance_mult to actual distance: 1.5->0.3m, 2.0->0.6m
            max_retreat = 0.3 + (self.current_action.distance_mult - 1.5) * 0.6  # Linear interpolation
            dy = -max_retreat * ease_t
            base_pos, base_orn = self.env.get_base_pose()
            new_y = self.initial_base_y + dy
            new_pos = [base_pos[0], new_y, base_pos[2]]
            p.resetBasePositionAndOrientation(self.env.spotId, new_pos, base_orn)
        
        # 3. Motion energy: Decays to minimal
        # Small settling oscillation that decays
        if progress > 0.7:
            # Settling motion at end
            settle_time = elapsed_time - (duration_s * 0.7)
            decay_factor = np.exp(-settle_time * 2.0)  # Exponential decay
            settle_offset = 0.03 * decay_factor * np.sin(settle_time * 3.0 * np.pi)
            settle_config = {
                'fl.hy': base_hy + settle_offset * 0.1,
                'fr.hy': base_hy + settle_offset * 0.1,
                'hl.hy': base_hy + 0.1 + settle_offset * 0.1,
                'hr.hy': base_hy + 0.1 + settle_offset * 0.1
            }
            self.env._apply_config(settle_config)
    
    def _render_caution(self, progress: float, elapsed_time: float, duration_s: float):
        """
        CAUTION: Contract implementation
        
        Physical signature (all required):
        - Posture: upright, slightly rigid stance
        - Distance: maintain OR slight retreat (≤ 0.2 m)
        - Motion energy: low but continuous
        - Attention: small left-right yaw scan (±5-10°)
        """
        # 1. Posture: Upright, slightly rigid (already set by pose, maintain it)
        # Re-apply caution pose to maintain
        self.env.apply_reaction_pose('caution')
        
        # 2. Distance: Maintain or slight retreat (≤0.2m)
        # Contract: maintain OR slight retreat ≤0.2m
        if self.current_action.distance_mult > 1.0 and self.current_action.distance_mult < 1.5:
            # Slight retreat (distance_mult 1.0-1.5 maps to 0.0-0.2m)
            retreat_progress = min(1.0, progress / 0.8)
            ease_t = retreat_progress * retreat_progress * (3.0 - 2.0 * retreat_progress)
            # Map: 1.0->0m, 1.1->0.04m, 1.2->0.08m, etc. up to 1.5->0.2m
            max_retreat = (self.current_action.distance_mult - 1.0) * 0.4  # Max 0.2m at 1.5
            dy = -max_retreat * ease_t
            base_pos, base_orn = self.env.get_base_pose()
            new_y = self.initial_base_y + dy
            new_pos = [base_pos[0], new_y, base_pos[2]]
            p.resetBasePositionAndOrientation(self.env.spotId, new_pos, base_orn)
        
        # 3. Motion energy: Low but continuous
        # Small rhythmic micro-motion
        micro_motion = 0.02 * np.sin(elapsed_time * 1.5 * np.pi)
        micro_config = {
            'fl.hy': 0.5 + micro_motion * 0.1,
            'fr.hy': 0.5 + micro_motion * 0.1,
        }
        self.env._apply_config(micro_config)
        
        # 4. Attention: Left-right yaw scan (±5-10°)
        # Convert 5-10 degrees to radians: ±0.087 to ±0.175 rad
        scan_amplitude = 0.12  # ~7 degrees in radians
        yaw_angle = scan_amplitude * np.sin(elapsed_time * 1.0 * np.pi)
        self.env.apply_yaw_scan(yaw_angle)
    
    def _render_check_in(self, progress: float, elapsed_time: float, duration_s: float):
        """
        CHECK_IN: Contract implementation
        
        Physical signature (all required):
        - Posture: slight forward lean (no base translation)
        - Distance: maintained
        - Motion energy: gentle rhythmic micro-motion (breathing/sway)
        - Attention: forward-biased, stable
        """
        # 1. Posture: Slight forward lean (already set by pose)
        # Re-apply check-in pose to maintain forward lean
        self.env.apply_reaction_pose('check-in')
        
        # 2. Distance: Maintained (no base movement)
        # (No distance change)
        
        # 3. Motion energy: Gentle rhythmic micro-motion
        # Breathing/sway pattern: slower, gentler than other motions
        breath_phase = elapsed_time * 0.8 * np.pi  # Slower frequency
        breath_offset = 0.04 * np.sin(breath_phase)  # Gentle amplitude
        
        # Apply as subtle height variation (breathing)
        breath_config = {
            'fl.hy': 0.7 + breath_offset * 0.15,
            'fr.hy': 0.7 + breath_offset * 0.15,
            'hl.hy': 0.7 + breath_offset * 0.15,
            'hr.hy': 0.7 + breath_offset * 0.15
        }
        self.env._apply_config(breath_config)
        
        # 4. Attention: Forward-biased, stable (no yaw scanning)
        # (No yaw changes)
    
    def _render_engage(self, progress: float, elapsed_time: float, duration_s: float):
        """
        ENGAGE: Contract implementation
        
        Physical signature (all required):
        - Posture: tall, open stance with forward lean
        - Distance: small smooth approach (0.15-0.3 m, only if safe)
        - Motion energy: rhythmic, higher amplitude (bounce)
        - Attention: forward and stable
        """
        # 1. Posture: Tall, open stance with forward lean (already set by pose)
        # Re-apply engage pose to maintain
        self.env.apply_reaction_pose('engage')
        
        # 2. Distance: Small smooth approach (0.15-0.3 m, only if safe)
        # Contract: approach 0.15-0.3m only if safe (distance_mult < 1.0)
        if self.current_action.distance_mult < 1.0 and self.current_action.speed_mult > 0.5:
            approach_progress = min(1.0, progress / 0.8)
            ease_t = approach_progress * approach_progress * (3.0 - 2.0 * approach_progress)
            # Map distance_mult: 0.7-0.9 -> 0.15-0.3m approach
            # Linear: 0.7->0.3m, 0.8->0.22m, 0.9->0.15m (inverse: lower mult = more approach)
            approach_dist = 0.3 - (0.7 - self.current_action.distance_mult) * 0.3  # 0.7->0.3m, 0.8->0.22m, 0.9->0.15m
            dy = approach_dist * ease_t
            base_pos, base_orn = self.env.get_base_pose()
            new_y = self.initial_base_y + dy
            new_pos = [base_pos[0], new_y, base_pos[2]]
            p.resetBasePositionAndOrientation(self.env.spotId, new_pos, base_orn)
        
        # 3. Motion energy: Rhythmic, higher amplitude (bounce)
        bounce_phase = elapsed_time * 2.5 * np.pi  # Higher frequency
        bounce_offset = 0.08 * np.sin(bounce_phase)  # Higher amplitude
        
        # Apply as up/down body height oscillation
        bounce_config = {
            'fl.hy': 0.4 + bounce_offset * 0.2,
            'fr.hy': 0.4 + bounce_offset * 0.2,
            'hl.hy': 0.8 + bounce_offset * 0.2,
            'hr.hy': 0.8 + bounce_offset * 0.2
        }
        self.env._apply_config(bounce_config)
        
        # 4. Attention: Forward and stable (no yaw scanning)
        # (No yaw changes)
    
    def _render_neutral(self, progress: float, elapsed_time: float, duration_s: float):
        """
        NEUTRAL: Contract implementation
        
        Physical signature:
        - Posture: normal stand
        - Motion: minimal idle motion only
        - Distance: unchanged
        """
        # 1. Posture: Normal stand (maintain)
        self.env.stand()
        
        # 2. Motion: Minimal idle motion only
        # Very subtle, slow motion
        idle_phase = elapsed_time * 0.3 * np.pi  # Very slow
        idle_offset = 0.01 * np.sin(idle_phase)  # Very small amplitude
        
        idle_config = {
            'fl.hy': 0.6 + idle_offset * 0.05,
            'fr.hy': 0.6 + idle_offset * 0.05,
        }
        self.env._apply_config(idle_config)
        
        # 3. Distance: Unchanged
        # (No distance change)
    
    def hold_with_animation(self, duration_s: float, steps_per_second: int = 60):
        """
        Hold current action with animation for specified duration.
        
        This integrates all physical variables (posture, distance, motion energy, attention)
        into a time-extended behavior.
        
        Args:
            duration_s: Duration to hold in seconds
            steps_per_second: Simulation steps per second
        """
        # Legacy helper kept for backwards compatibility; prefer external high-rate stepping.
        total_steps = int(duration_s * steps_per_second)
        for _ in range(total_steps):
            self.env.step()
    
    def maintain_pose(self):
        """Re-apply target pose to maintain it (call during hold)."""
        if self.current_action is None:
            return
        # Pose is maintained by apply_animation which re-applies poses
