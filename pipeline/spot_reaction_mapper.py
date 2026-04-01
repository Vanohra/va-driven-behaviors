"""
Spot Reaction Mapper Module

This module maps emotion-based reaction recommendations to actual Boston Dynamics Spot robot commands.
It provides a high-level interface for controlling Spot's behavior based on emotional state analysis.
"""

from typing import Dict, Optional, Tuple
from enum import Enum
from .reaction_action import ReactionAction


class SpotCommand(Enum):
    """Enumeration of Spot robot command types."""
    STOP = "stop"
    MOVE_BACKWARD = "move_backward"
    MOVE_FORWARD = "move_forward"
    SLOW_DOWN = "slow_down"
    NORMAL_SPEED = "normal_speed"
    INCREASE_DISTANCE = "increase_distance"
    DECREASE_DISTANCE = "decrease_distance"
    MAINTAIN_DISTANCE = "maintain_distance"
    CALM_LED = "calm_led"
    PLAYFUL_LED = "playful_led"
    NEUTRAL_LED = "neutral_led"
    CALM_AUDIO = "calm_audio"
    ENCOURAGING_AUDIO = "encouraging_audio"
    GENTLE_TONE = "gentle_tone"
    PAUSE = "pause"
    CONTINUE = "continue"
    SIT = "sit"
    STAND = "stand"


class SpotReactionMapper:
    """
    Maps emotion reaction recommendations to Spot robot commands.
    
    This class parses reaction recommendation strings and generates appropriate
    Spot robot commands based on the emotional state and recommended behavior.
    """
    
    def __init__(self, confidence_threshold: float = 0.6, volatility_high: float = 0.25, volatility_med: float = 0.15):
        """Initialize the reaction mapper."""
        self.current_speed = 1.0  # Normal speed multiplier
        self.current_distance = 1.0  # Distance multiplier
        self.current_state = "standing"  # Current robot state
        self.confidence_threshold = confidence_threshold
        self.volatility_high = volatility_high
        self.volatility_med = volatility_med
    
    def map_to_action(self, 
                     va_label: str,
                     trends: Dict,
                     volatility: float,
                     confidence: float,
                     valence: float,
                     arousal: float) -> Tuple[ReactionAction, str]:
        """
        Map emotion analysis to structured ReactionAction.
        
        Args:
            va_label: VA state label (e.g., 'negative-high-arousal')
            trends: Dict with 'valence_direction', 'arousal_direction', etc.
            volatility: Combined volatility metric (max of valence/arousal volatility)
            confidence: Confidence in the emotion detection (0.0 to 1.0)
            valence: Mean valence value
            arousal: Mean arousal value
        
        Returns:
            Tuple of (ReactionAction, explain_string)
        """
        # Determine volatility level
        is_high_vol = volatility >= self.volatility_high
        is_med_vol = volatility >= self.volatility_med and not is_high_vol
        
        # Determine if stable and confident enough for VA_POSE
        is_stable = not is_high_vol and not is_med_vol
        use_va_pose = is_stable and confidence >= self.confidence_threshold
        
        # High volatility: DE_ESCALATE
        if is_high_vol:
            intent = "DE_ESCALATE"
            speed_mult = 0.0 if volatility >= self.volatility_high * 1.2 else 0.3
            distance_mult = 2.0  # Maps to 0.45m retreat (middle of 0.3-0.6m range)
            pose_mode = "REACTION_POSE"
            pose_name = "DE_ESCALATE"
            duration_s = 2.2  # Contract: 1.8-2.5s, use middle
            explain = f"High volatility detected (vol={volatility:.3f}): de-escalating"
        
        # Medium volatility: CAUTION
        elif is_med_vol:
            intent = "CAUTION"
            speed_mult = 0.3
            distance_mult = 1.1  # Maps to ≤0.2m retreat per contract
            pose_mode = "REACTION_POSE"
            pose_name = "CAUTION"
            duration_s = 1.75  # Contract: 1.5-2.0s, use middle
            explain = f"Medium volatility (vol={volatility:.3f}): exercising caution"
        
        # Stable: determine by VA state
        else:
            va_lower = va_label.lower()
            
            # Negative-high-arousal -> DE_ESCALATE
            if 'negative' in va_lower and 'high-arousal' in va_lower:
                intent = "DE_ESCALATE"
                speed_mult = 0.3
                distance_mult = 1.8 if arousal > 0.5 else 2.0  # More retreat for higher arousal
                pose_mode = "VA_POSE" if use_va_pose else "REACTION_POSE"
                pose_name = "DE_ESCALATE" if not use_va_pose else "VA_POSE"
                duration_s = 2.2  # Contract: 1.8-2.5s
                explain = f"Negative-high-arousal state: de-escalating (V={valence:.2f}, A={arousal:.2f})"
            
            # Negative-low-arousal -> CHECK_IN
            elif 'negative' in va_lower and 'low-arousal' in va_lower:
                intent = "CHECK_IN"
                speed_mult = 0.5  # Gentle motion energy
                distance_mult = 1.0  # Contract: distance maintained
                pose_mode = "VA_POSE" if use_va_pose else "REACTION_POSE"
                pose_name = "CHECK_IN" if not use_va_pose else "VA_POSE"
                duration_s = 1.75  # Contract: 1.5-2.0s
                explain = f"Negative-low-arousal state: checking in (V={valence:.2f}, A={arousal:.2f})"
            
            # Positive-high-arousal -> ENGAGE
            elif 'positive' in va_lower and 'high-arousal' in va_lower:
                intent = "ENGAGE"
                speed_mult = 0.9 if arousal > 0.5 else 0.7  # Higher energy
                # Only approach if safe (low volatility, high confidence)
                distance_mult = 0.8 if (arousal > 0.5 and not is_med_vol and confidence > 0.7) else 1.0
                pose_mode = "VA_POSE" if use_va_pose else "REACTION_POSE"
                pose_name = "ENGAGE" if not use_va_pose else "VA_POSE"
                duration_s = 1.5  # Contract: 1.2-1.8s, use middle
                explain = f"Positive-high-arousal state: engaging (V={valence:.2f}, A={arousal:.2f})"
            
            # Neutral -> NEUTRAL
            elif va_lower == 'neutral' or ('neutral' in va_lower and 'valence' in va_lower and 'arousal' in va_lower):
                intent = "NEUTRAL"
                speed_mult = 0.5  # Minimal motion energy
                distance_mult = 1.0  # Contract: distance unchanged
                pose_mode = "VA_POSE" if use_va_pose else "REACTION_POSE"
                pose_name = "NEUTRAL" if not use_va_pose else "VA_POSE"
                duration_s = 1.5  # Can be subtle per contract
                explain = f"Neutral state: maintaining (V={valence:.2f}, A={arousal:.2f})"
            
            # High-arousal (neutral valence) -> CAUTION
            elif 'high-arousal' in va_lower:
                intent = "CAUTION"
                speed_mult = 0.4
                distance_mult = 1.1  # Contract: maintain or ≤0.2m retreat
                pose_mode = "REACTION_POSE"
                pose_name = "CAUTION"
                duration_s = 1.75  # Contract: 1.5-2.0s
                explain = f"High-arousal state: caution (V={valence:.2f}, A={arousal:.2f})"
            
            # Low-arousal (neutral valence) -> CHECK_IN
            elif 'low-arousal' in va_lower:
                intent = "CHECK_IN"
                speed_mult = 0.5
                distance_mult = 1.0  # Contract: distance maintained
                pose_mode = "VA_POSE" if use_va_pose else "REACTION_POSE"
                pose_name = "CHECK_IN" if not use_va_pose else "VA_POSE"
                duration_s = 1.75  # Contract: 1.5-2.0s
                explain = f"Low-arousal state: checking in (V={valence:.2f}, A={arousal:.2f})"
            
            # Fallback: NEUTRAL
            else:
                intent = "NEUTRAL"
                speed_mult = 0.5
                distance_mult = 1.0  # Contract: distance unchanged
                pose_mode = "REACTION_POSE"
                pose_name = "NEUTRAL"
                duration_s = 1.5
                explain = f"Unknown state '{va_label}': neutral behavior (V={valence:.2f}, A={arousal:.2f})"
        
        # Build debug dict
        debug = {
            'volatility': volatility,
            'confidence': confidence,
            'va_label': va_label,
            'valence': valence,
            'arousal': arousal,
            'valence_direction': trends.get('valence_direction', 'unknown'),
            'arousal_direction': trends.get('arousal_direction', 'unknown'),
            'volatility_high_threshold': self.volatility_high,
            'volatility_med_threshold': self.volatility_med,
            'confidence_threshold': self.confidence_threshold
        }
        
        action = ReactionAction(
            intent=intent,
            speed_mult=speed_mult,
            distance_mult=distance_mult,
            pose_mode=pose_mode,
            pose_name=pose_name,
            duration_s=duration_s,
            explain=explain,
            debug=debug
        )
        
        return action, explain
    
    def parse_reaction(self, reaction_recommendation: str) -> Dict:
        """
        Parse a reaction recommendation string and extract command components.
        
        DEPRECATED: This method is kept for backward compatibility.
        New code should use map_to_action() with structured inputs.
        
        Args:
            reaction_recommendation: String with reaction recommendation
        
        Returns:
            Dictionary with parsed command components:
            {
                'primary_action': str,
                'speed': float,
                'distance': str,
                'led_mode': str,
                'audio_mode': str,
                'movement': str
            }
        """
        reaction_lower = reaction_recommendation.lower()
        
        # Initialize command components
        commands = {
            'primary_action': 'continue',
            'speed': 1.0,
            'distance': 'maintain',
            'led_mode': 'neutral',
            'audio_mode': 'neutral',
            'movement': 'normal'
        }
        
        # Parse primary actions
        if 'stop immediately' in reaction_lower or 'stop' in reaction_lower:
            commands['primary_action'] = 'stop'
            commands['speed'] = 0.0
        elif 'pause' in reaction_lower or 'assess' in reaction_lower:
            commands['primary_action'] = 'pause'
            commands['speed'] = 0.0
        elif 'slow down' in reaction_lower or 'slow movement' in reaction_lower:
            commands['primary_action'] = 'slow_down'
            commands['speed'] = 0.3
        elif 'maintain' in reaction_lower and 'interaction' in reaction_lower:
            commands['primary_action'] = 'continue'
            commands['speed'] = 1.0
        elif 'engage' in reaction_lower:
            commands['primary_action'] = 'continue'
            commands['speed'] = 1.0
        elif 'continue' in reaction_lower:
            commands['primary_action'] = 'continue'
            commands['speed'] = 1.0
        
        # Parse distance commands
        if 'increase distance significantly' in reaction_lower:
            commands['distance'] = 'increase_significant'
        elif 'increase distance' in reaction_lower:
            commands['distance'] = 'increase'
        elif 'keep distance' in reaction_lower or 'maintain distance' in reaction_lower:
            commands['distance'] = 'maintain'
        elif 'reduce distance' in reaction_lower or 'decrease distance' in reaction_lower:
            commands['distance'] = 'decrease'
        
        # Parse LED/visual feedback
        if 'calm voice/led' in reaction_lower or 'calm led' in reaction_lower:
            commands['led_mode'] = 'calm'
        elif 'playful' in reaction_lower or 'encouraging' in reaction_lower:
            commands['led_mode'] = 'playful'
        elif 'positive tone' in reaction_lower:
            commands['led_mode'] = 'positive'
        else:
            commands['led_mode'] = 'neutral'
        
        # Parse audio feedback
        if 'calm voice' in reaction_lower or 'calm tone' in reaction_lower:
            commands['audio_mode'] = 'calm'
        elif 'gentle tone' in reaction_lower:
            commands['audio_mode'] = 'gentle'
        elif 'playful' in reaction_lower or 'encouraging tone' in reaction_lower:
            commands['audio_mode'] = 'encouraging'
        elif 'positive tone' in reaction_lower:
            commands['audio_mode'] = 'positive'
        else:
            commands['audio_mode'] = 'neutral'
        
        # Parse movement style
        if 'minimize movement' in reaction_lower:
            commands['movement'] = 'minimal'
        elif 'normal speed' in reaction_lower:
            commands['movement'] = 'normal'
        elif 'slow' in reaction_lower:
            commands['movement'] = 'slow'
        
        return commands
    
    def map_to_spot_commands(self, reaction_recommendation: str, 
                            notes: str = "",
                            va_state_label: str = "") -> Dict:
        """
        Map reaction recommendation to Spot robot commands.
        
        Args:
            reaction_recommendation: String with reaction recommendation
            notes: Additional notes about the recommendation
            va_state_label: VA state label (e.g., 'negative-high-arousal')
        
        Returns:
            Dictionary with Spot commands:
            {
                'commands': list of SpotCommand enums,
                'speed_multiplier': float,
                'distance_action': str,
                'led_mode': str,
                'audio_mode': str,
                'movement_style': str,
                'parsed_components': dict
            }
        """
        parsed = self.parse_reaction(reaction_recommendation)
        
        commands = []
        
        # Map primary actions to commands
        if parsed['primary_action'] == 'stop':
            commands.append(SpotCommand.STOP)
        elif parsed['primary_action'] == 'pause':
            commands.append(SpotCommand.PAUSE)
        elif parsed['primary_action'] == 'slow_down':
            commands.append(SpotCommand.SLOW_DOWN)
        elif parsed['primary_action'] == 'continue':
            commands.append(SpotCommand.CONTINUE)
        
        # Map distance actions
        if parsed['distance'] == 'increase_significant':
            commands.append(SpotCommand.INCREASE_DISTANCE)
            commands.append(SpotCommand.MOVE_BACKWARD)
        elif parsed['distance'] == 'increase':
            commands.append(SpotCommand.INCREASE_DISTANCE)
        elif parsed['distance'] == 'decrease':
            commands.append(SpotCommand.DECREASE_DISTANCE)
        else:
            commands.append(SpotCommand.MAINTAIN_DISTANCE)
        
        # Map LED modes
        if parsed['led_mode'] == 'calm':
            commands.append(SpotCommand.CALM_LED)
        elif parsed['led_mode'] == 'playful':
            commands.append(SpotCommand.PLAYFUL_LED)
        else:
            commands.append(SpotCommand.NEUTRAL_LED)
        
        # Map audio modes
        if parsed['audio_mode'] == 'calm':
            commands.append(SpotCommand.CALM_AUDIO)
        elif parsed['audio_mode'] == 'gentle':
            commands.append(SpotCommand.GENTLE_TONE)
        elif parsed['audio_mode'] == 'encouraging':
            commands.append(SpotCommand.ENCOURAGING_AUDIO)
        else:
            # Default neutral audio
            pass
        
        # Update internal state
        self.current_speed = parsed['speed']
        if parsed['distance'] in ['increase', 'increase_significant']:
            self.current_distance = min(2.0, self.current_distance + 0.2)
        elif parsed['distance'] == 'decrease':
            self.current_distance = max(0.5, self.current_distance - 0.2)
        
        return {
            'commands': commands,
            'speed_multiplier': parsed['speed'],
            'distance_action': parsed['distance'],
            'led_mode': parsed['led_mode'],
            'audio_mode': parsed['audio_mode'],
            'movement_style': parsed['movement'],
            'parsed_components': parsed,
            'reaction_recommendation': reaction_recommendation,
            'notes': notes,
            'va_state_label': va_state_label
        }
    
    def get_command_summary(self, command_dict: Dict) -> str:
        """
        Generate a human-readable summary of Spot commands.
        
        Args:
            command_dict: Dictionary from map_to_spot_commands
        
        Returns:
            Human-readable string summary
        """
        summary_parts = []
        
        # Primary actions
        if command_dict['speed_multiplier'] == 0.0:
            summary_parts.append("STOP")
        elif command_dict['speed_multiplier'] < 0.5:
            summary_parts.append("SLOW DOWN")
        else:
            summary_parts.append("CONTINUE")
        
        # Distance
        if command_dict['distance_action'] == 'increase_significant':
            summary_parts.append("MOVE BACK SIGNIFICANTLY")
        elif command_dict['distance_action'] == 'increase':
            summary_parts.append("INCREASE DISTANCE")
        elif command_dict['distance_action'] == 'decrease':
            summary_parts.append("DECREASE DISTANCE")
        else:
            summary_parts.append("MAINTAIN DISTANCE")
        
        # LED
        if command_dict['led_mode'] != 'neutral':
            summary_parts.append(f"LED: {command_dict['led_mode'].upper()}")
        
        # Audio
        if command_dict['audio_mode'] != 'neutral':
            summary_parts.append(f"AUDIO: {command_dict['audio_mode'].upper()}")
        
        return " | ".join(summary_parts)


def create_spot_command_executor(env):
    """
    Create a function that executes Spot commands on a SpotEnv instance.
    
    Args:
        env: SpotEnv instance
    
    Returns:
        Function that takes command_dict and executes commands
    """
    def execute_commands(command_dict: Dict):
        """
        Execute Spot commands from a command dictionary.
        
        Args:
            command_dict: Dictionary from SpotReactionMapper.map_to_spot_commands
        """
        speed_mult = command_dict['speed_multiplier']
        distance_action = command_dict['distance_action']
        led_mode = command_dict['led_mode']
        audio_mode = command_dict['audio_mode']
        movement_style = command_dict['movement_style']
        
        # Execute primary action
        if speed_mult == 0.0:
            # Stop - use sit or minimal movement pose
            if hasattr(env, 'sit'):
                env.sit()
        elif speed_mult < 0.5:
            # Slow down - adjust pose to be more cautious
            if hasattr(env, 'set_speed'):
                env.set_speed(speed_mult)
        else:
            # Normal speed
            if hasattr(env, 'set_speed'):
                env.set_speed(1.0)
        
        # Execute distance action (if env supports it)
        if hasattr(env, 'adjust_distance'):
            if distance_action == 'increase_significant':
                env.adjust_distance(2.0)  # Move back significantly
            elif distance_action == 'increase':
                env.adjust_distance(1.5)  # Move back moderately
            elif distance_action == 'decrease':
                env.adjust_distance(0.7)  # Move closer
            else:
                env.adjust_distance(1.0)  # Maintain
        
        # LED and audio modes would be handled by external systems
        # For now, we just log them
        if led_mode != 'neutral':
            print(f"[LED] Setting mode: {led_mode}")
        
        if audio_mode != 'neutral':
            print(f"[AUDIO] Setting mode: {audio_mode}")
        
        return True
    
    return execute_commands
