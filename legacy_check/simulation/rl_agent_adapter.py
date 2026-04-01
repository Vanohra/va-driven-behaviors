"""
RL Agent Adapter for Emotion Pipeline

This adapter uses a pretrained ARS agent to generate intelligent, balanced locomotion
while expressing emotions. The agent maintains balance and adjusts leg angles to prevent
tipping over, making emotional expressions look more realistic.

Key features:
- Loads pretrained ARS agent (e.g., Agent 2229)
- Maps Valence/Arousal to locomotion parameters (desired_velocity, desired_rate, StepLength, YawRate)
- Uses agent's policy to generate balanced actions
- Blends emotional poses with agent-generated locomotion
"""

import time
import numpy as np
import pybullet as p
from typing import Optional, Dict, Tuple
import sys
import os
import copy

# Add paths for imports
# Calculate absolute path to spotmicro directory
_current_file = os.path.abspath(__file__)
# From: spot_mini_mini/spot_bullet/src/emotion/integration/rl_agent_adapter.py
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

# Add spot_bullet/src to path so ars_lib can be found
# From emotion/integration/ -> go up 2 levels to get to spot_bullet/src/
spot_bullet_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if spot_bullet_src_path not in sys.path:
    sys.path.insert(0, spot_bullet_src_path)

# Debug: print path if needed
# print(f"[RLAgentAdapter] Added to path: {spot_bullet_src_path}")
# print(f"[RLAgentAdapter] ars_lib exists: {os.path.exists(os.path.join(spot_bullet_src_path, 'ars_lib'))}")

from spotmicro.OpenLoopSM.SpotOL import BezierStepper
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.Kinematics.SpotKinematics import SpotModel

# Import ARS agent components (ars_lib is in spot_bullet/src/ars_lib)
from ars_lib.ars import ARSAgent, Normalizer, Policy

# ARS agent constants
CD_SCALE = 0.05
RESIDUALS_SCALE = 0.015
Z_SCALE = 0.05
alpha = 0.7
actions_to_filter = 14
P_yaw = 5.0


class RLAgentAdapter:
    """
    Adapter that uses pretrained ARS agent for intelligent locomotion with emotions.
    
    Maps Valence/Arousal to:
    - desired_velocity: Forward/backward speed (from Arousal)
    - desired_rate: Yaw rotation rate (from Valence, negative = left, positive = right)
    - StepLength: Step length for BezierStepper (from Arousal)
    - YawRate: Direct yaw control (from Valence)
    """
    
    def __init__(self, new_env, agent_num: int = 2229, use_contacts: bool = True):
        """
        Initialize RL agent adapter.
        
        Args:
            new_env: spotBezierEnv instance
            agent_num: Agent number to load (default: 2229)
            use_contacts: Whether to use contact sensing (affects model path)
        """
        self.env = new_env
        self.spot = new_env.spot
        self._pybullet_client = new_env._pybullet_client
        self.quadruped = new_env.spot.quadruped
        
        # Initialize SpotModel, BezierStepper, BezierGait
        self.spot_model = SpotModel()
        self.bezier_stepper = BezierStepper(dt=new_env._time_step, mode=0)
        self.bezier_gait = BezierGait(dt=new_env._time_step)
        
        # Initialize ARS agent
        state_dim = new_env.observation_space.shape[0]
        action_dim = new_env.action_space.shape[0]
        
        normalizer = Normalizer(state_dim)
        policy = Policy(state_dim, action_dim)
        
        self.agent = ARSAgent(
            normalizer, 
            policy, 
            new_env, 
            self.bezier_stepper, 
            self.bezier_gait, 
            self.spot_model, 
            gui=False
        )
        
        # Load pretrained agent
        # Calculate path: from emotion/integration/ -> go up to spot_bullet/ -> models/
        my_path = os.path.abspath(os.path.dirname(__file__))
        # From emotion/integration/ -> ../../ -> spot_bullet/src/ -> ../ -> spot_bullet/ -> models/
        spot_bullet_dir = os.path.abspath(os.path.join(my_path, "../../.."))
        if use_contacts:
            models_path = os.path.join(spot_bullet_dir, "models", "contact")
        else:
            models_path = os.path.join(spot_bullet_dir, "models", "no_contact")
        
        file_name = "spot_ars_"
        agent_path = os.path.join(models_path, file_name + str(agent_num) + "_policy")
        
        if os.path.exists(agent_path):
            print(f"[RLAgentAdapter] Loading pretrained agent {agent_num} from {agent_path}")
            self.agent.load(os.path.join(models_path, file_name + str(agent_num)))
            self.agent.policy.episode_steps = np.inf
        else:
            print(f"[RLAgentAdapter] WARNING: Agent {agent_num} not found at {agent_path}")
            print(f"[RLAgentAdapter] Available agents in {models_path}:")
            if os.path.exists(models_path):
                agent_files = [f for f in os.listdir(models_path) if f.startswith(file_name) and f.endswith("_policy")]
                if agent_files:
                    agent_nums = [f.replace(file_name, "").replace("_policy", "") for f in agent_files]
                    print(f"[RLAgentAdapter]   Try: {', '.join(sorted(agent_nums, key=int)[:10])}...")
            print(f"[RLAgentAdapter] Using untrained agent (will not work well)")
        
        # Emotion-to-locomotion mapping state
        self.current_valence = 0.0  # Range: [-1, 1], negative = negative emotion, positive = positive
        self.current_arousal = 0.0  # Range: [0, 1], intensity of emotion
        self.desired_velocity = 0.0  # Forward/backward velocity
        self.desired_rate = 0.0  # Yaw rotation rate
        
        # Locomotion state
        self.T_bf = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_b0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.old_act = np.zeros(actions_to_filter)
        
        # Emotional pose blending (optional - for static poses)
        self.emotional_pose_active = False
        self.emotional_pose_joint_targets: Dict[str, float] = {}
        self.verbose = True  # Enable verbose output
        
        # Intent-based pose mode state
        self.current_intent: Optional[str] = None
        self.current_pose_mode: Optional[str] = None  # "REACTION_POSE" or "VA_POSE"
        self.current_confidence: float = 0.0
        
        # Baseline poses (same as NewSimSpotAdapter for consistency)
        self.baseline_poses: Dict[str, Dict[str, float]] = {
            "NEUTRAL": {
                'fl.hx': 0.0, 'fl.hy': 0.6, 'fl.kn': -1.4,
                'fr.hx': 0.0, 'fr.hy': 0.6, 'fr.kn': -1.4,
                'hl.hx': 0.0, 'hl.hy': 0.6, 'hl.kn': -1.4,
                'hr.hx': 0.0, 'hr.hy': 0.6, 'hr.kn': -1.4,
            },
            "DE_ESCALATE": {
                # Lower body height (crouch): lower hy values, more bent knees
                # Backward lean: front legs extended (higher hy), back legs bent (lower hy)
                # Enhanced backward lean: increased difference between front and back
                'fl.hx': 0.0, 'fl.hy': 0.9, 'fl.kn': -1.4,   # Front left: more extended for evident backward lean
                'fr.hx': 0.0, 'fr.hy': 0.9, 'fr.kn': -1.4,   # Front right: more extended for evident backward lean
                'hl.hx': 0.0, 'hl.hy': 0.4, 'hl.kn': -1.0,   # Back left: more bent, lower (stronger crouch + backward lean)
                'hr.hx': 0.0, 'hr.hy': 0.4, 'hr.kn': -1.0,   # Back right: more bent, lower (stronger crouch + backward lean)
            },
            "CHECK_IN": {
                'fl.hx': 0.0, 'fl.hy': 0.8, 'fl.kn': -1.8,
                'fr.hx': 0.0, 'fr.hy': 0.8, 'fr.kn': -1.8,
                'hl.hx': 0.0, 'hl.hy': 0.4, 'hl.kn': -1.0,
                'hr.hx': 0.0, 'hr.hy': 0.4, 'hr.kn': -1.0,
            },
            "ENGAGE": {
                'fl.hx': 0.0, 'fl.hy': 0.3, 'fl.kn': -0.8,
                'fr.hx': 0.0, 'fr.hy': 0.3, 'fr.kn': -0.8,
                'hl.hx': 0.0, 'hl.hy': 0.8, 'hl.kn': -1.8,
                'hr.hx': 0.0, 'hr.hy': 0.8, 'hr.kn': -1.8,
            },
            "CAUTION": {
                # Fully upright posture: tall stance with straight legs (no crouch)
                # Subtle backward bias: front legs slightly higher than back for backing away
                'fl.hx': 0.0, 'fl.hy': 0.85, 'fl.kn': -1.3,   # Front left: fully upright, straighter
                'fr.hx': 0.0, 'fr.hy': 0.85, 'fr.kn': -1.3,   # Front right: fully upright, straighter
                'hl.hx': 0.0, 'hl.hy': 0.8, 'hl.kn': -1.3,   # Back left: fully upright, slightly lower (backing away)
                'hr.hx': 0.0, 'hr.hy': 0.8, 'hr.kn': -1.3,   # Back right: fully upright, slightly lower (backing away)
            },
        }
        
        # Current locomotion measurements
        self._current_step_length = 0.0
        self._current_yaw_rate = 0.0
        self._current_step_velocity = 0.3
        
        # Time tracking for CAUTION yaw scanning
        self._caution_scan_time = 0.0
        
        # Build joint name mapping (for compatibility with old interface)
        self._build_joint_mapping()
        
        print("[RLAgentAdapter] Initialized with pretrained ARS agent")
    
    def _build_joint_mapping(self):
        """Build mapping from old joint names to new motor indices."""
        self.joints: Dict[str, int] = {}
        self._old_to_new_mapping: Dict[str, str] = {}
        self._new_to_old_mapping: Dict[str, str] = {}
        
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
        
        if hasattr(self.spot, '_joint_name_to_id') and self.spot._joint_name_to_id:
            for old_name, new_name in mapping.items():
                if new_name in self.spot._joint_name_to_id:
                    joint_idx = self.spot._joint_name_to_id[new_name]
                    self.joints[old_name] = joint_idx
                    self._old_to_new_mapping[old_name] = new_name
                    self._new_to_old_mapping[new_name] = old_name
    
    def set_emotion(self, valence: float, arousal: float, 
                   volatility: float = 0.0, 
                   trends: Optional[Dict] = None,
                   intent: Optional[str] = None,
                   pose_mode: Optional[str] = None,
                   confidence: float = 1.0,
                   speed_mult: float = 1.0,
                   distance_mult: float = 1.0):
        """
        Set emotion values and map to locomotion parameters.
        
        Args:
            valence: [-1, 1], negative = negative emotion, positive = positive emotion
            arousal: [0, 1], intensity of emotion
            volatility: [0, 1], volatility of emotion (affects speed/intensity)
            trends: Optional dict with 'valence_direction', 'arousal_direction', 'valence_delta', 'arousal_delta'
            intent: Optional intent string (DE_ESCALATE, CAUTION, CHECK_IN, ENGAGE, NEUTRAL)
            pose_mode: Optional pose mode ("REACTION_POSE" or "VA_POSE")
            confidence: Confidence value [0, 1]
            speed_mult: Speed multiplier from intent (0.0 to 1.0)
            distance_mult: Distance multiplier from intent (typically 0.7, 1.0, 1.5, or 2.0)
        """
        self.current_valence = np.clip(valence, -1.0, 1.0)
        self.current_arousal = np.clip(arousal, 0.0, 1.0)
        self.current_volatility = np.clip(volatility, 0.0, 1.0)
        self.current_intent = intent
        self.current_pose_mode = pose_mode
        self.current_confidence = np.clip(confidence, 0.0, 1.0)
        
        # If in REACTION_POSE mode, use intent-based discrete behavior
        if pose_mode == "REACTION_POSE" and intent:
            # Reset scan time when switching away from CAUTION
            if intent.upper() != "CAUTION":
                self._caution_scan_time = 0.0
            self._apply_reaction_pose_mode(intent, speed_mult, distance_mult)
        # If in VA_POSE mode, use continuous VA modulation
        elif pose_mode == "VA_POSE":
            self._apply_va_pose_mode(valence, arousal, volatility, trends, speed_mult, distance_mult)
        # Default: use original behavior (for backward compatibility)
        else:
            self._apply_default_emotion_mapping(valence, arousal, volatility, trends)
    
    def _apply_reaction_pose_mode(self, intent: str, speed_mult: float, distance_mult: float):
        """
        Apply REACTION_POSE mode: discrete, intent-based behavior.
        
        This triggers a predefined stable pose based on intent.
        For REACTION_POSE mode, we prioritize static poses with minimal locomotion
        to avoid "stomping in place" behavior.
        """
        # Set baseline pose based on intent
        intent_upper = intent.upper()
        if intent_upper in self.baseline_poses:
            self.set_emotional_pose(self.baseline_poses[intent_upper])
            if self.verbose:
                print(f"[RLAgentAdapter] REACTION_POSE mode: {intent_upper}")
        
        # Intent-specific locomotion parameters (designed to minimize stomping)
        # Format: (step_velocity_base, step_length_base, yaw_rate_base, description)
        intent_params = {
            "DE_ESCALATE": {
                "step_velocity": 0.10,  # Very slow - minimal movement (reduced from 0.15)
                "step_length": -0.015,  # Small backward translation (reduced from -0.02 for slower retreat)
                "yaw_rate": 0.0,        # No rotation - stable orientation
                "description": "Crouched retreat pose with backward lean, minimal motion"
            },
            "CAUTION": {
                "step_velocity": 0.12,   # Very slow - much slower movement while backing away
                "step_length": -0.012,   # Slow backward retreat
                "yaw_rate": 0.0,        # Will be set dynamically for scanning
                "description": "Fully upright pose, very slow retreat with yaw scanning"
            },
            "CHECK_IN": {
                "step_velocity": 0.25,  # Moderate - gentle movement
                "step_length": 0.0,     # Hold position
                "yaw_rate": 0.0,        # No rotation
                "description": "Gentle static pose"
            },
            "ENGAGE": {
                "step_velocity": 0.4,   # Moderate - energetic but controlled
                "step_length": 0.01,    # Slight approach
                "yaw_rate": 0.0,        # No rotation
                "description": "Energetic but stable pose"
            },
            "NEUTRAL": {
                "step_velocity": 0.12,  # Very slow - minimal movement
                "step_length": 0.0,     # Hold position
                "yaw_rate": 0.0,        # No rotation
                "description": "Neutral static pose"
            },
        }
        
        # Get intent-specific parameters
        params = intent_params.get(intent_upper, {
            "step_velocity": 0.15,
            "step_length": 0.0,
            "yaw_rate": 0.0,
            "description": "Default static pose"
        })
        
        # Apply speed multiplier (but keep it low for REACTION_POSE)
        # Speed multiplier scales the base step velocity
        final_step_velocity = params["step_velocity"] * speed_mult
        
        # Special handling for DE_ESCALATE: even more restrictive clamping for minimal motion
        if intent_upper == "DE_ESCALATE":
            # Clamp to very low range for minimal micro-movements
            final_step_velocity = np.clip(final_step_velocity, 0.08, 0.15)
        elif intent_upper == "CAUTION":
            # Clamp to very low range for very slow movement while backing away
            final_step_velocity = np.clip(final_step_velocity, 0.08, 0.18)
        else:
            # Clamp to very low range to prevent stomping
            final_step_velocity = np.clip(final_step_velocity, 0.1, 0.5)
        
        # Apply distance multiplier to step length
        # distance_mult > 1.0 = retreat, < 1.0 = approach
        base_step_length = params["step_length"]
        if distance_mult > 1.0:
            # Retreat: make step length more negative
            final_step_length = base_step_length - abs(base_step_length) * (distance_mult - 1.0) * 0.5
        elif distance_mult < 1.0:
            # Approach: make step length more positive
            final_step_length = base_step_length + abs(base_step_length) * (1.0 - distance_mult) * 0.5
        else:
            final_step_length = base_step_length
        
        # Special handling for DE_ESCALATE: even more restrictive clamping for minimal translation
        if intent_upper == "DE_ESCALATE":
            # Clamp to very small range for minimal backward translation
            final_step_length = np.clip(final_step_length, -0.015, 0.0)  # Only allow backward movement
        elif intent_upper == "CAUTION":
            # Clamp to allow slow backward retreat while scanning
            final_step_length = np.clip(final_step_length, -0.015, 0.0)  # Only allow backward movement
        else:
            # Clamp step length to very small range to prevent translation
            final_step_length = np.clip(final_step_length, -0.02, 0.02)
        
        # Yaw rate (typically zero for static poses)
        # Note: Actual yaw scanning is added in step() method for CAUTION
        final_yaw_rate = params["yaw_rate"]
        
        # Update BezierStepper parameters
        self.bezier_stepper.StepLength = final_step_length
        self.bezier_stepper.YawRate = final_yaw_rate
        self.bezier_stepper.StepVelocity = final_step_velocity
        
        # Store for measurement
        self.desired_velocity = final_step_length * 10.0  # Convert to m/s estimate
        self.desired_rate = final_yaw_rate
        self._current_step_length = final_step_length
        self._current_yaw_rate = final_yaw_rate
        self._current_step_velocity = final_step_velocity
        
        if self.verbose:
            print(f"[RLAgentAdapter] {params['description']}: "
                  f"StepVel={final_step_velocity:.3f}, "
                  f"StepLen={final_step_length:.4f}, "
                  f"YawRate={final_yaw_rate:.3f}")
    
    def _apply_va_pose_mode(self, valence: float, arousal: float, volatility: float,
                           trends: Optional[Dict], speed_mult: float, distance_mult: float):
        """
        Apply VA_POSE mode: continuous modulation based on valence/arousal.
        
        In this mode we continuously modulate:
        - speed multiplier
        - distance multiplier (approach/retreat)
        - duration
        - yaw behavior
        - posture (body height, lean)
        
        For VA_POSE, we allow more movement than REACTION_POSE but still regulate
        to prevent excessive stomping.
        """
        # Base locomotion parameters from valence/arousal
        # Valence → forward/back lean + approach/retreat
        # Arousal → body height + motion energy
        
        # Apply distance multiplier: > 1.0 = retreat, < 1.0 = approach
        base_velocity = valence * arousal * 0.3  # Reduced from 0.5 to 0.3 to prevent stomping
        if distance_mult > 1.0:
            # Retreat: make velocity more negative
            base_velocity = base_velocity - abs(base_velocity) * (distance_mult - 1.0) * 0.3
        elif distance_mult < 1.0:
            # Approach: make velocity more positive
            base_velocity = base_velocity + abs(base_velocity) * (1.0 - distance_mult) * 0.3
        
        base_rate = -valence * arousal * 0.3  # Reduced from 0.5 to 0.3
        
        # Apply speed multiplier to step velocity (more regulated)
        base_step_velocity = 0.2 + arousal * 0.4  # Reduced range: 0.2 to 0.6
        base_step_velocity *= speed_mult
        
        # Apply volatility modulation (but less aggressive than REACTION_POSE)
        volatility_mult = 1.0 + volatility * 0.2  # Reduced from 0.3 to 0.2
        volatility_jitter = (np.random.random() - 0.5) * volatility * 0.05  # Reduced jitter
        
        # Apply trend-based adjustments
        if trends:
            if trends.get('valence_direction') == 'increasing':
                base_velocity *= (1.0 + abs(trends.get('valence_delta', 0.0)) * 0.15)  # Reduced
            if trends.get('arousal_direction') == 'increasing':
                base_step_velocity *= (1.0 + abs(trends.get('arousal_delta', 0.0)) * 0.2)  # Reduced
        
        # Combine all factors
        self.desired_velocity = base_velocity * volatility_mult + volatility_jitter
        self.desired_rate = base_rate * volatility_mult + volatility_jitter * 0.2  # Reduced
        
        # Update BezierStepper parameters with tighter limits
        step_length = self.desired_velocity * 0.1
        # Clamp to smaller range to prevent excessive movement
        self.bezier_stepper.StepLength = np.clip(step_length, -0.03, 0.03)  # Reduced from 0.05
        self.bezier_stepper.YawRate = np.clip(self.desired_rate, -0.5, 0.5)  # Reduced from 1.0
        # Clamp step velocity to prevent stomping
        self.bezier_stepper.StepVelocity = np.clip(base_step_velocity * volatility_mult, 0.15, 0.7)  # Reduced max
        
        # For VA_POSE, we also modulate posture through emotional pose blending
        # Map valence to forward/back lean, arousal to body height
        # This is done through subtle joint angle adjustments
        lean_factor = valence * 0.1  # Small lean adjustment
        height_factor = arousal * 0.15  # Body height adjustment
        
        # Create subtle pose modulation (blended with locomotion)
        va_pose_targets = {}
        for leg in ['fl', 'fr', 'hl', 'hr']:
            # Forward lean: front legs lower, back legs higher (positive valence)
            # Back lean: front legs higher, back legs lower (negative valence)
            if leg.startswith('f'):  # Front legs
                va_pose_targets[f'{leg}.hy'] = 0.6 - lean_factor  # Lower for forward lean
            else:  # Back legs
                va_pose_targets[f'{leg}.hy'] = 0.6 + lean_factor  # Higher for forward lean
            
            # Body height: higher arousal = taller stance
            va_pose_targets[f'{leg}.kn'] = -1.4 + height_factor  # Straighter legs for higher arousal
        
        # Blend VA pose with locomotion (lighter blend for VA_POSE)
        self.emotional_pose_active = True
        self.emotional_pose_joint_targets = va_pose_targets
        
        # Store for measurement
        self._current_step_length = step_length
        self._current_yaw_rate = self.bezier_stepper.YawRate
        self._current_step_velocity = self.bezier_stepper.StepVelocity
    
    def _apply_default_emotion_mapping(self, valence: float, arousal: float,
                                      volatility: float, trends: Optional[Dict]):
        """
        Apply default emotion mapping (original behavior for backward compatibility).
        """
        # Base locomotion parameters from valence/arousal
        base_velocity = valence * arousal * 0.5  # Max 0.5 m/s
        base_rate = -valence * arousal * 0.5  # Max 0.5 rad/s
        base_step_velocity = 0.3 + arousal * 0.5  # 0.3 to 0.8
        
        # Apply volatility modulation
        volatility_mult = 1.0 + volatility * 0.5  # 1.0 to 1.5x
        volatility_jitter = (np.random.random() - 0.5) * volatility * 0.2
        
        # Apply trend-based adjustments
        trend_mult = 1.0
        if trends:
            if trends.get('valence_direction') == 'increasing':
                trend_mult = 1.0 + abs(trends.get('valence_delta', 0.0)) * 0.3
            if trends.get('arousal_direction') == 'increasing':
                base_step_velocity *= (1.0 + abs(trends.get('arousal_delta', 0.0)) * 0.5)
        
        # Combine all factors
        self.desired_velocity = base_velocity * volatility_mult + volatility_jitter
        self.desired_rate = base_rate * volatility_mult + volatility_jitter * 0.5
        
        # Update BezierStepper parameters
        step_length = self.desired_velocity * 0.1
        self.bezier_stepper.StepLength = np.clip(step_length, -0.05, 0.05)
        self.bezier_stepper.YawRate = np.clip(self.desired_rate, -1.0, 1.0)
        self.bezier_stepper.StepVelocity = np.clip(base_step_velocity * volatility_mult, 0.1, 1.5)
        
        # Store for measurement
        self._current_step_length = step_length
        self._current_yaw_rate = self.bezier_stepper.YawRate
        self._current_step_velocity = self.bezier_stepper.StepVelocity
    
    def set_emotional_pose(self, joint_targets: Dict[str, float]):
        """
        Set emotional pose targets (for blending with locomotion).
        
        Args:
            joint_targets: Dictionary mapping old joint names (fl.hx, etc.) to target angles
        """
        self.emotional_pose_active = True
        self.emotional_pose_joint_targets = joint_targets.copy()
    
    def clear_emotional_pose(self):
        """Clear emotional pose (return to pure locomotion)."""
        self.emotional_pose_active = False
        self.emotional_pose_joint_targets = {}
    
    def step(self) -> Tuple[bool, Optional[str]]:
        """
        Step the simulation using RL agent.
        
        Returns:
            (done, info): done indicates if episode ended, info is optional message
        """
        # Get BezierStepper parameters
        pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = \
            self.bezier_stepper.StateMachine()
        
        # Update environment with external observations
        self.env.spot.GetExternalObservations(self.bezier_gait, self.bezier_stepper)
        
        # Get current state
        state = self.env.return_state()
        
        # Normalize state
        self.agent.normalizer.observe(state)
        state = self.agent.normalizer.normalize(state)
        
        # Get action from policy
        action = self.agent.policy.evaluate(state, delta=None, direction=None)
        
        # Apply exponential filter to actions
        action = np.tanh(action)
        action[:actions_to_filter] = alpha * self.old_act + (1.0 - alpha) * action[:actions_to_filter]
        self.old_act = action[:actions_to_filter].copy()
        
        # Modify BezierStepper parameters based on agent action
        ClearanceHeight += action[0] * CD_SCALE
        
        # Clip all parameters
        StepLength = np.clip(StepLength, self.bezier_stepper.StepLength_LIMITS[0],
                            self.bezier_stepper.StepLength_LIMITS[1])
        StepVelocity = np.clip(StepVelocity, self.bezier_stepper.StepVelocity_LIMITS[0],
                              self.bezier_stepper.StepVelocity_LIMITS[1])
        LateralFraction = np.clip(LateralFraction, self.bezier_stepper.LateralFraction_LIMITS[0],
                                 self.bezier_stepper.LateralFraction_LIMITS[1])
        YawRate = np.clip(YawRate, self.bezier_stepper.YawRate_LIMITS[0],
                         self.bezier_stepper.YawRate_LIMITS[1])
        ClearanceHeight = np.clip(ClearanceHeight, self.bezier_stepper.ClearanceHeight_LIMITS[0],
                                 self.bezier_stepper.ClearanceHeight_LIMITS[1])
        PenetrationDepth = np.clip(PenetrationDepth, self.bezier_stepper.PenetrationDepth_LIMITS[0],
                                  self.bezier_stepper.PenetrationDepth_LIMITS[1])
        
        # Add slow yaw scanning for CAUTION intent (left-right while backing away)
        # Do this BEFORE auto yaw correction so scanning takes priority
        if self.current_intent == "CAUTION" and self.current_pose_mode == "REACTION_POSE":
            # Slow yaw scan: oscillate left-right at ~0.35 Hz (~2.86 second period) - slower
            # Amplitude: ±0.3 rad/s (~17.2 degrees/second) - more visible scanning
            scan_frequency = 0.35  # Hz (slower, more deliberate)
            scan_amplitude = 0.3  # rad/s (increased for visibility)
            dt = 1.0 / 240.0  # Physics timestep
            self._caution_scan_time += dt
            scan_yaw_rate = scan_amplitude * np.sin(2.0 * np.pi * scan_frequency * self._caution_scan_time)
            # Set scanning yaw rate directly (override any other yaw rate)
            YawRate = scan_yaw_rate  # Set directly instead of adding, so scanning is primary
            # Also update bezier_stepper directly to ensure it's used
            self.bezier_stepper.YawRate = scan_yaw_rate
            if self.verbose and int(self._caution_scan_time * 240) % 600 == 0:  # Print every ~2.5 seconds
                print(f"[CAUTION] Yaw scan: {scan_yaw_rate:.3f} rad/s ({np.degrees(scan_yaw_rate):.1f} deg/s), time: {self._caution_scan_time:.2f}s")
        else:
            # Auto yaw correction (only when not scanning)
            yaw = self.env.return_yaw()
            YawRate += -yaw * P_yaw
        
        # Get contact information
        contacts = state[-4:] if len(state) >= 4 else [0, 0, 0, 0]
        
        # Generate foot trajectories using BezierGait
        T_bf = self.bezier_gait.GenerateTrajectory(
            StepLength, LateralFraction, YawRate, StepVelocity,
            self.T_b0, self.T_bf, ClearanceHeight, PenetrationDepth, contacts
        )
        
        # Apply agent's residual actions to foot positions
        action[2:] *= RESIDUALS_SCALE
        T_bf_copy = copy.deepcopy(T_bf)
        T_bf_copy["FL"][:3, 3] += action[2:5]
        T_bf_copy["FR"][:3, 3] += action[5:8]
        T_bf_copy["BL"][:3, 3] += action[8:11]
        T_bf_copy["BR"][:3, 3] += action[11:14]
        
        # Adjust body height
        pos[2] += abs(action[1]) * Z_SCALE
        
        # Convert foot positions to joint angles using IK
        joint_angles = self.spot_model.IK(orn, pos, T_bf_copy)
        
        # Blend with emotional pose if active
        # For VA_POSE mode, use lighter blending; for REACTION_POSE, use stronger blending
        # For CAUTION, use stronger blending to make upright posture and stiffening visible
        if self.emotional_pose_active and self.emotional_pose_joint_targets:
            if self.current_pose_mode == "VA_POSE":
                joint_angles = self._blend_emotional_pose(joint_angles, blend_factor=0.2)  # Lighter blend
            elif self.current_intent == "CAUTION":
                joint_angles = self._blend_emotional_pose(joint_angles, blend_factor=0.4)  # Stronger blend for visible upright posture
            else:
                joint_angles = self._blend_emotional_pose(joint_angles, blend_factor=0.3)  # Standard blend
        
        # Pass joint angles to environment
        self.env.pass_joint_angles(joint_angles.reshape(-1))
        
        # Step environment
        action_array = np.zeros(self.env.action_space.shape[0])
        next_state, reward, done, _ = self.env.step(action_array)
        
        # Update T_bf for next iteration
        self.T_bf = T_bf
        
        return done, None
    
    def _blend_emotional_pose(self, locomotion_joint_angles: np.ndarray, blend_factor: float = 0.3) -> np.ndarray:
        """
        Blend emotional pose with locomotion joint angles.
        
        Args:
            locomotion_joint_angles: Joint angles from IK (4x3 array)
            blend_factor: Blending factor (0.0 = pure locomotion, 1.0 = pure emotional pose)
            
        Returns:
            Blended joint angles
        """
        # Convert locomotion angles to dictionary format
        joint_order = ['FL', 'FR', 'BL', 'BR']
        angle_names = ['hx', 'hy', 'kn']  # hip x, hip y, knee
        
        blended = locomotion_joint_angles.copy()
        
        # Blend emotional pose targets with locomotion
        for old_name, target_angle in self.emotional_pose_joint_targets.items():
            if old_name in self._old_to_new_mapping:
                # Extract leg and joint from old name (e.g., 'fl.hy' -> 'FL', 'hy')
                parts = old_name.split('.')
                if len(parts) == 2:
                    leg_abbrev = parts[0].upper()  # 'fl' -> 'FL'
                    joint_name = parts[1]  # 'hy'
                    
                    # Map to joint_order index
                    leg_map = {'FL': 0, 'FR': 1, 'BL': 2, 'BR': 3}
                    if leg_abbrev in leg_map:
                        leg_idx = leg_map[leg_abbrev]
                        if joint_name in angle_names:
                            joint_idx = angle_names.index(joint_name)
                            # Blend: (1 - blend_factor) * locomotion + blend_factor * emotional pose
                            blended[leg_idx, joint_idx] = (
                                (1.0 - blend_factor) * blended[leg_idx, joint_idx] + 
                                blend_factor * target_angle
                            )
        
        return blended
    
    # Compatibility methods for SpotEnv interface
    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get base position and orientation."""
        pos = self.spot.GetBasePosition()
        orn = self.spot.GetBaseOrientation()
        return np.array(pos), np.array(orn)
    
    def move_base_smooth(self, dy: float, duration_s: float, steps: int = None):
        """
        Move base smoothly (maps to emotion-based locomotion).
        
        Args:
            dy: Distance to move in Y direction
            duration_s: Duration (ignored, uses emotion mapping)
            steps: Number of steps (ignored)
        """
        # Map movement to arousal/valence
        # Positive dy = forward = positive valence
        # Negative dy = backward = negative valence
        valence = np.sign(dy) if abs(dy) > 0.01 else 0.0
        arousal = min(abs(dy) * 2.0, 1.0)  # Scale to [0, 1]
        self.set_emotion(valence, arousal)
    
    def hold(self, duration_s: float, steps_per_second: int = 240):
        """Hold current pose (stop locomotion)."""
        self.set_emotion(0.0, 0.0)  # No movement
        steps = int(duration_s * steps_per_second)
        for _ in range(steps):
            self.step()
            time.sleep(1.0 / steps_per_second)
    
    def apply_reaction_pose(self, reaction_type: str):
        """Apply reaction pose (for compatibility - uses emotional pose system)."""
        # This would map reaction types to joint targets
        # For now, just clear emotional pose
        self.clear_emotional_pose()
    
    def set_baseline_pose(self, pose_name: str):
        """
        Set baseline pose (for compatibility with BehaviorRenderer).
        
        This applies BOTH the physical pose (joint angles) AND emotion-based locomotion.
        
        Args:
            pose_name: Name of pose (e.g., "NEUTRAL", "DE_ESCALATE", "CHECK_IN", etc.)
        """
        pose_name_upper = pose_name.upper()
        
        # Apply physical pose if available
        if pose_name_upper in self.baseline_poses:
            self.set_emotional_pose(self.baseline_poses[pose_name_upper])
            if self.verbose:
                print(f"[RLAgentAdapter] Applied physical pose: {pose_name_upper}")
        else:
            # Clear emotional pose if pose not found
            self.clear_emotional_pose()
        
        # Also map pose to emotion-based locomotion for combined effect
        pose_to_emotion = {
            "NEUTRAL": (0.0, 0.0),
            "DE_ESCALATE": (-0.3, 0.2),  # Slight negative valence, low arousal
            "CHECK_IN": (0.2, 0.3),  # Positive valence, moderate arousal
            "ENGAGE": (0.5, 0.6),  # Positive valence, high arousal
            "CAUTION": (-0.2, 0.5),  # Slight negative valence, moderate arousal
        }
        
        if pose_name_upper in pose_to_emotion:
            valence, arousal = pose_to_emotion[pose_name_upper]
            self.set_emotion(valence, arousal)
        else:
            # Default to neutral
            self.set_emotion(0.0, 0.0)
    
    def emotional_pose(self, valence: float, arousal: float):
        """
        Set emotional pose from valence/arousal (for compatibility with BehaviorRenderer).
        
        Args:
            valence: [-1, 1], negative = negative emotion, positive = positive emotion
            arousal: [0, 1], intensity of emotion
        """
        self.set_emotion(valence, arousal)
    
    def set_joint_position(self, joint_name: str, target_pos: float):
        """
        Set joint position (for compatibility - blends with locomotion).
        
        Args:
            joint_name: Old joint name (e.g., 'fl.hy')
            target_pos: Target angle in radians
        """
        # Add to emotional pose targets for blending
        if not self.emotional_pose_active:
            self.emotional_pose_active = True
        self.emotional_pose_joint_targets[joint_name] = target_pos
    
    def stand(self):
        """Stand pose (for compatibility)."""
        self.set_baseline_pose("NEUTRAL")
    
    def sit(self):
        """Sit pose (for compatibility)."""
        # Map sit to low arousal, neutral valence
        self.set_emotion(0.0, 0.1)
    
    def get_locomotion_state(self) -> Dict[str, float]:
        """
        Get current locomotion state measurements.
        
        Returns:
            Dictionary with current locomotion variables:
            - speed: Current forward/backward speed (m/s)
            - yaw_rate: Current yaw rotation rate (rad/s)
            - step_length: Current step length (m)
            - step_velocity: Current stepping velocity
            - valence: Current valence value
            - arousal: Current arousal value
            - volatility: Current volatility value
        """
        # Get actual base velocity from environment
        base_pos = self.spot.GetBasePosition()
        base_vel = self.spot.GetBaseVelocity() if hasattr(self.spot, 'GetBaseVelocity') else [0, 0, 0]
        
        return {
            'speed': self.desired_velocity,  # Desired forward/backward speed
            'yaw_rate': self._current_yaw_rate,  # Current yaw rotation rate
            'step_length': self._current_step_length,  # Current step length
            'step_velocity': self._current_step_velocity,  # Current stepping velocity
            'base_velocity_x': base_vel[0] if len(base_vel) > 0 else 0.0,
            'base_velocity_y': base_vel[1] if len(base_vel) > 1 else 0.0,
            'valence': self.current_valence,
            'arousal': self.current_arousal,
            'volatility': self.current_volatility,
        }
    
    def get_pose_state(self) -> Dict[str, any]:
        """
        Get current pose state.
        
        Returns:
            Dictionary with pose information:
            - active: Whether emotional pose is active
            - pose_name: Current baseline pose name (if set)
            - joint_targets: Current joint targets
        """
        return {
            'active': self.emotional_pose_active,
            'joint_targets': self.emotional_pose_joint_targets.copy(),
        }
    
    def reset(self):
        """Reset the environment and agent state."""
        self.env.reset(desired_velocity=self.desired_velocity, desired_rate=self.desired_rate)
        self.T_bf = copy.deepcopy(self.spot_model.WorldToFoot)
        self.T_b0 = copy.deepcopy(self.spot_model.WorldToFoot)
        self.bezier_gait.reset()
        self.bezier_stepper.reshuffle()
        self.old_act = np.zeros(actions_to_filter)
