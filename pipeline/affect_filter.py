"""
Affect Signal Filtering Module

Applies temporal filtering and rate limiting to valence/arousal signals
to prevent jittery behavior and rapid state transitions.
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque


class AffectFilter:
    """
    Filters affect signals (valence/arousal) to reduce noise and enforce
    smooth transitions.
    
    Features:
    - Exponential moving average (EMA) smoothing
    - Rate limiting on per-update changes
    - Hysteresis for state transitions
    """
    
    def __init__(self, 
                 ema_alpha: float = 0.10,  # Maximum smoothing (reduced from 0.15) - very smooth
                 max_delta_per_update: float = 0.05,  # Very restrictive (reduced from 0.07) - very gradual changes
                 hysteresis_margin: float = 0.1):
        """
        Initialize affect filter.
        
        Args:
            ema_alpha: EMA smoothing factor (0.0-1.0, lower = more smoothing)
            max_delta_per_update: Maximum allowed change per update (clamps velocity)
            hysteresis_margin: Margin for state transitions (prevents rapid flipping)
        """
        self.ema_alpha = ema_alpha
        self.max_delta_per_update = max_delta_per_update
        self.hysteresis_margin = hysteresis_margin
        
        # Filtered state
        self.filtered_valence: Optional[float] = None
        self.filtered_arousal: Optional[float] = None
        
        # For hysteresis tracking
        self.last_va_state: Optional[str] = None
        self.state_confidence: float = 0.0  # Accumulates evidence for state change
    
    def update(self, valence: float, arousal: float) -> Tuple[float, float]:
        """
        Update filter with new raw valence/arousal values.
        
        Args:
            valence: Raw valence (-1 to 1)
            arousal: Raw arousal (-1 to 1)
        
        Returns:
            Tuple of (filtered_valence, filtered_arousal)
        """
        # Initialize if first update
        if self.filtered_valence is None:
            self.filtered_valence = valence
            self.filtered_arousal = arousal
            return (valence, arousal)
        
        # EMA smoothing
        filtered_v = self.ema_alpha * valence + (1.0 - self.ema_alpha) * self.filtered_valence
        filtered_a = self.ema_alpha * arousal + (1.0 - self.ema_alpha) * self.filtered_arousal
        
        # Rate limiting: clamp per-update change
        delta_v = filtered_v - self.filtered_valence
        delta_a = filtered_a - self.filtered_arousal
        
        if abs(delta_v) > self.max_delta_per_update:
            delta_v = np.sign(delta_v) * self.max_delta_per_update
        if abs(delta_a) > self.max_delta_per_update:
            delta_a = np.sign(delta_a) * self.max_delta_per_update
        
        # Apply rate-limited deltas
        self.filtered_valence += delta_v
        self.filtered_arousal += delta_a
        
        # Clamp to valid range
        self.filtered_valence = np.clip(self.filtered_valence, -1.0, 1.0)
        self.filtered_arousal = np.clip(self.filtered_arousal, -1.0, 1.0)
        
        return (self.filtered_valence, self.filtered_arousal)
    
    def should_transition_state(self, new_va_state: str) -> bool:
        """
        Check if state transition should occur (with hysteresis).
        
        Args:
            new_va_state: Proposed new VA state label
        
        Returns:
            True if transition should occur, False if should maintain current state
        """
        if self.last_va_state is None:
            self.last_va_state = new_va_state
            return True
        
        if new_va_state == self.last_va_state:
            # Same state: reset confidence
            self.state_confidence = 0.0
            return False
        
        # Different state: accumulate confidence
        self.state_confidence += 0.2  # Increment per update
        
        # Require sustained evidence before transitioning
        threshold = 0.6  # Need 3-4 consistent updates
        if self.state_confidence >= threshold:
            self.last_va_state = new_va_state
            self.state_confidence = 0.0
            return True
        
        return False
    
    def reset(self):
        """Reset filter state (e.g., on new session)."""
        self.filtered_valence = None
        self.filtered_arousal = None
        self.last_va_state = None
        self.state_confidence = 0.0
