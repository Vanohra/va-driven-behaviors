"""
VA Pipeline — Core Analysis Package

Provides the full VA (valence-arousal) analysis stack:
  - Robust preprocessing (winsorize, Hampel filter, EMA smoothing)
  - Session baseline estimation (trimmed mean)
  - Trend analysis (Theil-Sen slope + start/end delta)
  - Volatility estimation (MAD)
  - Percentile-based VA state classification
  - Intent selection (5-intent policy)
  - Reaction mapping (ReactionAction)
  - AffectFilter (EMA + rate limiting + hysteresis)

All functions are stateless and work on numpy arrays.
None of this code depends on PyBullet, gym, or simulation.
"""

from .emotion_analyzer import (
    load_calibration,
    analyze_emotion_stream,
    preprocess_series,
    compute_robust_baseline,
    compute_trends,
    compute_state_label,
)
from .robust_stats import mad, iqr, trimmed_mean, winsorize, hampel_filter
from .reaction_action import ReactionAction
from .spot_reaction_mapper import SpotReactionMapper
from .intent_selector import IntentSelector, Intent
from .affect_filter import AffectFilter

__all__ = [
    # Calibration & analysis
    "load_calibration",
    "analyze_emotion_stream",
    "preprocess_series",
    "compute_robust_baseline",
    "compute_trends",
    "compute_state_label",
    # Robust stats
    "mad", "iqr", "trimmed_mean", "winsorize", "hampel_filter",
    # Reaction / policy
    "ReactionAction",
    "SpotReactionMapper",
    "IntentSelector",
    "Intent",
    # Signal filtering
    "AffectFilter",
]
