"""
ReactionAction Schema Module

Defines the structured action object that replaces prose string parsing.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Literal


@dataclass
class ReactionAction:
    """
    Structured action object for robot reactions.
    
    This replaces fragile string parsing with a deterministic schema.
    """
    intent: Literal["DE_ESCALATE", "CAUTION", "CHECK_IN", "ENGAGE", "NEUTRAL"]
    speed_mult: float  # 0.0 to 1.0
    distance_mult: float  # Typically 0.7, 1.0, 1.5, or 2.0 (continuous allowed)
    pose_mode: Literal["REACTION_POSE", "VA_POSE"]
    pose_name: str  # e.g., "DE_ESCALATE", "HAPPY_PRESET", etc.
    duration_s: float  # Default 1.5-2.0 seconds
    explain: str  # Human-readable explanation (for logging/UI only, never parsed)
    debug: Dict  # Store volatility, confidence, trends, thresholds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReactionAction':
        """Create from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"ReactionAction(intent={self.intent}, speed={self.speed_mult:.2f}, "
                f"distance={self.distance_mult:.2f}, pose_mode={self.pose_mode}, "
                f"duration={self.duration_s:.2f}s)")
