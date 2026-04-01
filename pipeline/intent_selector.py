"""
Intent Selection Module

Implements the 5-intent emotion policy logic that selects one of:
- DE_ESCALATE: back off, calm situation
- CAUTION: alert, careful
- CHECK_IN: supportive, present
- ENGAGE: playful / interactive
- NEUTRAL: maintain normal behavior

This module does NOT modify the VA pipeline - it only uses the outputs.
"""

from typing import Dict, Tuple, Literal
from enum import Enum


class Intent(Enum):
    """The 5 emotion policy intents."""
    DE_ESCALATE = "DE_ESCALATE"
    CAUTION = "CAUTION"
    CHECK_IN = "CHECK_IN"
    ENGAGE = "ENGAGE"
    NEUTRAL = "NEUTRAL"


class IntentSelector:
    """
    Selects intent from VA pipeline outputs (valence, arousal, volatility, confidence).
    
    Logic:
    1. Check volatility: High → DE_ESCALATE, Medium → CAUTION, Low → check VA state
    2. If low volatility, use VA label to determine intent
    3. Confidence gating: low confidence → REACTION_POSE, high confidence + low volatility → VA_POSE
    """
    
    def __init__(self, 
                 volatility_high_threshold: float = 0.25,
                 volatility_med_threshold: float = 0.15,
                 confidence_threshold: float = 0.6):
        """
        Initialize intent selector.
        
        Args:
            volatility_high_threshold: Threshold for high volatility (default: 0.25)
            volatility_med_threshold: Threshold for medium volatility (default: 0.15)
            confidence_threshold: Threshold for high confidence (default: 0.6)
        """
        self.volatility_high_threshold = volatility_high_threshold
        self.volatility_med_threshold = volatility_med_threshold
        self.confidence_threshold = confidence_threshold
    
    def select_intent(self,
                     valence: float,
                     arousal: float,
                     volatility: float,
                     confidence: float,
                     va_label: str) -> Tuple[Intent, Literal["REACTION_POSE", "VA_POSE"], str]:
        """
        Select intent and pose mode from VA pipeline outputs.
        
        Args:
            valence: Valence value from VA pipeline
            arousal: Arousal value from VA pipeline
            volatility: Volatility value from VA pipeline
            confidence: Confidence value from VA pipeline
            va_label: VA state label (e.g., 'negative-high-arousal', 'neutral', etc.)
        
        Returns:
            Tuple of (intent, pose_mode, explanation)
        """
        # Step 1: Check volatility
        is_high_vol = volatility >= self.volatility_high_threshold
        is_med_vol = volatility >= self.volatility_med_threshold and not is_high_vol
        is_low_vol = not is_high_vol and not is_med_vol
        
        # High volatility → DE_ESCALATE
        if is_high_vol:
            intent = Intent.DE_ESCALATE
            pose_mode = "REACTION_POSE"
            explain = f"High volatility detected (vol={volatility:.3f}): de-escalating"
            return intent, pose_mode, explain
        
        # Medium volatility → CAUTION
        if is_med_vol:
            intent = Intent.CAUTION
            pose_mode = "REACTION_POSE"
            explain = f"Medium volatility (vol={volatility:.3f}): exercising caution"
            return intent, pose_mode, explain
        
        # Step 2: Low volatility - use VA label to determine intent
        va_lower = va_label.lower()
        
        # Negative-high-arousal → DE_ESCALATE
        if 'negative' in va_lower and 'high-arousal' in va_lower:
            intent = Intent.DE_ESCALATE
            explain = f"Negative-high-arousal state: de-escalating (V={valence:.2f}, A={arousal:.2f})"
        
        # Negative-low-arousal → CHECK_IN
        elif 'negative' in va_lower and 'low-arousal' in va_lower:
            intent = Intent.CHECK_IN
            explain = f"Negative-low-arousal state: checking in (V={valence:.2f}, A={arousal:.2f})"
        
        # Positive-high-arousal → ENGAGE
        elif 'positive' in va_lower and 'high-arousal' in va_lower:
            intent = Intent.ENGAGE
            explain = f"Positive-high-arousal state: engaging (V={valence:.2f}, A={arousal:.2f})"
        
        # Neutral → NEUTRAL
        elif va_lower == 'neutral' or ('neutral' in va_lower and 'valence' in va_lower and 'arousal' in va_lower):
            intent = Intent.NEUTRAL
            explain = f"Neutral state: maintaining (V={valence:.2f}, A={arousal:.2f})"
        
        # High-arousal (ambiguous) → CAUTION
        elif 'high-arousal' in va_lower:
            intent = Intent.CAUTION
            explain = f"High-arousal state (ambiguous): caution (V={valence:.2f}, A={arousal:.2f})"
        
        # Low-arousal (ambiguous) → CHECK_IN
        elif 'low-arousal' in va_lower:
            intent = Intent.CHECK_IN
            explain = f"Low-arousal state (ambiguous): checking in (V={valence:.2f}, A={arousal:.2f})"
        
        # Fallback: NEUTRAL
        else:
            intent = Intent.NEUTRAL
            explain = f"Unknown state '{va_label}': neutral behavior (V={valence:.2f}, A={arousal:.2f})"
        
        # Step 3: Confidence gating
        # If confidence is low → force REACTION_POSE
        # If confidence is high and volatility is low → allow VA_POSE
        if confidence < self.confidence_threshold:
            pose_mode = "REACTION_POSE"
            explain += f" (low confidence={confidence:.2f}, using REACTION_POSE)"
        elif is_low_vol and confidence >= self.confidence_threshold:
            pose_mode = "VA_POSE"
            explain += f" (high confidence={confidence:.2f}, low volatility={volatility:.3f}, using VA_POSE)"
        else:
            pose_mode = "REACTION_POSE"
            explain += f" (using REACTION_POSE)"
        
        return intent, pose_mode, explain
    
    def get_pose_name(self, intent: Intent, pose_mode: Literal["REACTION_POSE", "VA_POSE"]) -> str:
        """
        Get pose name for the given intent and pose mode.
        
        Args:
            intent: Selected intent
            pose_mode: Selected pose mode
        
        Returns:
            Pose name string
        """
        if pose_mode == "VA_POSE":
            return "VA_POSE"
        
        # For REACTION_POSE, return the intent name
        return intent.value
