"""
Window Analyzer

Runs the full VA pipeline on a single video/audio file (one captured window).
Used by both:
  - run_offline.py  (single video file analysis)
  - online_session.py  (each 10-second live capture window)

Reuses existing functions unchanged:
  test_emotions.extract_video_features / extract_audio_features / align_features / predict_emotions
  (test_emotions.py must be present in the project root)
  pipeline.emotion_analyzer.analyze_emotion_stream
  pipeline.spot_reaction_mapper.SpotReactionMapper.map_to_action
  pipeline.intent_selector.IntentSelector
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Core pipeline — no simulation dependencies
from pipeline.emotion_analyzer import analyze_emotion_stream
from pipeline.spot_reaction_mapper import SpotReactionMapper
from pipeline.intent_selector import IntentSelector


class WindowAnalyzer:
    """
    Runs the complete VA analysis pipeline on one video window.

    Holds the model and calibration so they are loaded only once per session.
    Call analyze_window(video_path, audio_path) for each clip.
    """

    def __init__(
        self,
        model,                       # Pre-loaded JointCAM model (from test_emotions.load_model)
        device: str,                 # 'cpu' or 'cuda'
        calibration: Optional[Dict], # From pipeline.load_calibration(), or None for fallback
        debug: bool = False,
    ):
        self.model       = model
        self.device      = device
        self.calibration = calibration
        self.debug       = debug

        self.reaction_mapper  = SpotReactionMapper()
        self.intent_selector  = IntentSelector(
            volatility_high_threshold=0.25,
            volatility_med_threshold=0.15,
            confidence_threshold=0.6,
        )

    def analyze_window(
        self,
        video_path: str,
        audio_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        last_v:      Optional[float] = None,
        last_a:      Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Run the full pipeline on one video clip.

        Args:
            video_path:  Path to video file (.mp4, .avi, etc.)
            audio_path:  Path to audio file (.wav, .mp4, etc.).
                         Defaults to video_path (audio extracted from video).

        Returns:
            Dict with all analyze_emotion_stream keys, plus:
                'reaction_action'      – ReactionAction instance
                'reaction_explanation' – human-readable string
                'volatility'           – max(valence_vol, arousal_vol)
            Returns None on any failure.
        """
        if audio_path is None:
            audio_path = video_path

        try:
            from test_emotions import (
                extract_video_features,
                extract_audio_features,
                align_features,
                predict_emotions,
            )
        except ImportError as e:
            print(
                f"[WindowAnalyzer] Cannot import test_emotions: {e}\n"
                "  Place test_emotions.py in the project root directory."
            )
            return None

        # ── Step 1: Feature extraction ──────────────────────────────────────
        try:
            if self.debug:
                print(f"  [analyzer] video features: {video_path}")
            video_features, fps = extract_video_features(
                video_path, self.device, num_samples=num_samples
            )
            audio_features       = extract_audio_features(audio_path)
        except Exception as e:
            print(f"[WindowAnalyzer] Feature extraction failed: {e}")
            if self.debug:
                import traceback; traceback.print_exc()
            return None

        # ── Step 2: Alignment ───────────────────────────────────────────────
        try:
            video_aligned, audio_aligned = align_features(video_features, audio_features)
        except Exception as e:
            print(f"[WindowAnalyzer] Alignment failed: {e}")
            return None

        # ── Step 3: Inference ───────────────────────────────────────────────
        try:
            valence_series, arousal_series = predict_emotions(
                self.model, video_aligned, audio_aligned, self.device
            )
            valence_series = np.asarray(valence_series).flatten()
            arousal_series = np.asarray(arousal_series).flatten()

            if self.debug:
                print(
                    f"  [analyzer] {len(valence_series)} frames  "
                    f"V=[{valence_series.min():.3f}, {valence_series.max():.3f}]  "
                    f"A=[{arousal_series.min():.3f}, {arousal_series.max():.3f}]"
                )
        except Exception as e:
            print(f"[WindowAnalyzer] Inference failed: {e}")
            if self.debug:
                import traceback; traceback.print_exc()
            return None

        if len(valence_series) < 2:
            print(f"[WindowAnalyzer] Only {len(valence_series)} frames — skipping.")
            return None

        # ── Step 4: Robust analysis (unchanged from offline pipeline) ────────
        try:
            analysis = analyze_emotion_stream(
                valence_series,
                arousal_series,
                calibration=self.calibration,
                debug=self.debug,
            )
        except Exception as e:
            print(f"[WindowAnalyzer] analyze_emotion_stream failed: {e}")
            if self.debug:
                import traceback; traceback.print_exc()
            return None

        if analysis is None:
            return None

        # ── Step 5: Reaction mapping ─────────────────────────────────────────
        va_label   = analysis.get("va_state_label", "neutral")
        volatility = max(
            analysis.get("valence_volatility", 0.0),
            analysis.get("arousal_volatility", 0.0),
        )
        trends = {
            "valence_direction": analysis.get("valence_direction", "stable"),
            "arousal_direction": analysis.get("arousal_direction", "stable"),
            "valence_delta":     analysis.get("valence_delta", 0.0),
            "arousal_delta":     analysis.get("arousal_delta", 0.0),
        }

        try:
            reaction_action, explanation = self.reaction_mapper.map_to_action(
                va_label=va_label,
                trends=trends,
                volatility=volatility,
                confidence=analysis.get("state_confidence", 0.5),
                valence=analysis.get("valence", 0.0),
                arousal=analysis.get("arousal", 0.0),
                last_v=last_v,
                last_a=last_a,
            )
        except Exception as e:
            print(f"[WindowAnalyzer] Reaction mapping failed: {e}")
            reaction_action = None
            explanation     = ""

        analysis["reaction_action"]      = reaction_action
        analysis["reaction_explanation"] = explanation
        analysis["volatility"]           = volatility

        return analysis
