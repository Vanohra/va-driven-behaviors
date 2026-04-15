"""
Online Session Manager

Orchestrates a configurable live interaction session (default 30 s).

Architecture:
  Session (30 s)
    ├── Window 0  t=0–10s:  capture → analyze → decide → print
    ├── Window 1  t=10–20s: capture → analyze → decide → print
    └── Window 2  t=20–30s: capture → analyze → decide → print

Stability-first design:
  - Full robust VA analysis (preprocess → baseline → trends → volatility →
    state classification → reaction mapping) on every window.
  - Between-window stability enforced by:
      1. Minimum confidence threshold (reject low-confidence estimates)
      2. Cooldown period (prevent flip-flopping between adjacent windows)
  - Graceful fallback to NEUTRAL on analysis failure.
  - No robot hardware needed — outputs go to terminal.
"""

import time
import os
from datetime import datetime
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# ReactionHistory — hysteresis for between-window behavior changes
# ─────────────────────────────────────────────────────────────────────────────

class ReactionHistory:
    """
    Applies two stability gates to proposed intent changes:

    1. Confidence gate  — reject if state_confidence < min_confidence
    2. Cooldown gate    — reject if < cooldown_s seconds since last change

    With only 3 windows in a 30-second session these two rules prevent
    a single noisy window from triggering an unwarranted behavior switch.
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        cooldown_s: float = 0.5,
        fallback_intent: str = "NEUTRAL",
    ):
        self.min_confidence = min_confidence
        self.cooldown_s     = cooldown_s
        self.fallback_intent = fallback_intent

        self._last_intent: Optional[str] = None
        self._last_change_time: float = 0.0
        self._history: List[Dict] = []

    def evaluate(
        self,
        proposed_intent: str,
        confidence: float,
        va_label: str,
        current_time: float,
    ) -> tuple:
        """
        Returns (effective_intent: str, reason: str, did_change: bool).

        reason codes:
            'first'          – first window, always accepted
            'low_confidence' – confidence below threshold, fallback used
            'cooldown'       – within cooldown, current intent maintained
            'same'           – proposed == current, no change
            'changed'        – accepted new intent
        """
        if self._last_intent is None:
            self._record(proposed_intent, confidence, va_label, current_time, changed=True)
            return proposed_intent, "first", True

        if confidence < self.min_confidence:
            self._record(self._last_intent, confidence, va_label, current_time)
            return self.fallback_intent, "low_confidence", False

        if proposed_intent == self._last_intent:
            self._record(self._last_intent, confidence, va_label, current_time)
            return self._last_intent, "same", False

        if (current_time - self._last_change_time) < self.cooldown_s:
            self._record(self._last_intent, confidence, va_label, current_time)
            return self._last_intent, "cooldown", False

        self._record(proposed_intent, confidence, va_label, current_time, changed=True)
        return proposed_intent, "changed", True

    def _record(self, intent, confidence, va_label, t, changed=False):
        if changed:
            self._last_change_time = t
        self._last_intent = intent
        self._history.append(
            {"time": t, "intent": intent, "confidence": confidence,
             "va_label": va_label, "changed": changed}
        )

    @property
    def current_intent(self) -> Optional[str]:
        return self._last_intent

    @property
    def history(self) -> List[Dict]:
        return self._history


# ─────────────────────────────────────────────────────────────────────────────
# OnlineSession
# ─────────────────────────────────────────────────────────────────────────────

class OnlineSession:
    """
    Manages one complete live VA interaction session.

    Usage (webcam must be open before calling run()):
        with LiveCapture() as capture:
            session = OnlineSession(analyzer, capture, ...)
            results = session.run()
    """

    def __init__(
        self,
        window_analyzer,
        live_capture,
        session_duration_s: float = 30.0,
        window_duration_s: float  = 10.0,
        min_confidence: float     = 0.55,
        cooldown_s: float         = 8.0,
        debug: bool               = False,
        cleanup_temp: bool        = True,
    ):
        self.analyzer     = window_analyzer
        self.capture      = live_capture
        self.session_dur  = session_duration_s
        self.window_dur   = window_duration_s
        self.debug        = debug
        self.cleanup_temp = cleanup_temp

        self.n_windows = max(1, int(session_duration_s / window_duration_s))

        self.history = ReactionHistory(
            min_confidence=min_confidence,
            cooldown_s=cooldown_s,
        )
        self.session_results: List[Dict] = []

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        """Run the full session. Blocks until session_duration_s elapses."""
        self._print_header()
        session_start = time.time()

        for idx in range(self.n_windows):
            t_offset = idx * self.window_dur
            self._print_window_banner(idx, t_offset)

            # 1. Capture
            capture_result = self._safe_capture(idx)
            if capture_result is None:
                self.session_results.append(
                    _error_result(idx, t_offset, "capture_failed")
                )
                continue

            # 2. Analyze
            analysis = self._safe_analyze(capture_result)

            # 3. Decide
            now = time.time()
            if analysis is not None:
                proposed = _extract_intent(analysis)
                conf     = analysis.get("state_confidence", 0.0)
                label    = analysis.get("va_state_label", "unknown")
            else:
                proposed = self.history.fallback_intent
                conf     = 0.0
                label    = "unknown"

            effective, reason, did_change = self.history.evaluate(
                proposed, conf, label, now
            )

            # 4. Print
            self._print_window_result(
                idx, analysis, effective, reason, did_change, capture_result
            )

            # 5. Store
            self.session_results.append({
                "window_idx":       idx,
                "window_start_s":   t_offset,
                "analysis":         analysis,
                "effective_intent": effective,
                "reason":           reason,
                "did_change":       did_change,
                "capture":          capture_result,
                "window_elapsed_s": time.time() - (session_start + t_offset),
            })

            # 6. Clean temp files
            if self.cleanup_temp and capture_result:
                from online.live_capture import LiveCapture
                LiveCapture.cleanup_files(capture_result)

        self._print_summary(time.time() - session_start)
        return self.session_results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _safe_capture(self, window_id: int) -> Optional[Dict]:
        try:
            return self.capture.capture_window(
                duration_s=self.window_dur, window_id=window_id
            )
        except Exception as e:
            print(f"  [ERROR] Capture: {e}")
            if self.debug:
                import traceback; traceback.print_exc()
            return None

    def _safe_analyze(self, capture_result: Dict) -> Optional[Dict]:
        print("  Running VA analysis pipeline...", flush=True)
        try:
            return self.analyzer.analyze_window(
                video_path=capture_result["video_path"],
                audio_path=capture_result["audio_path"],
            )
        except Exception as e:
            print(f"  [ERROR] Analysis: {e}")
            if self.debug:
                import traceback; traceback.print_exc()
            return None

    # ── Terminal output ───────────────────────────────────────────────────────

    def _print_header(self):
        w = 64
        print()
        print("╔" + "═" * w + "╗")
        print("║" + "  VA ONLINE SESSION  ".center(w) + "║")
        print("╚" + "═" * w + "╝")
        print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration:  {self.session_dur:.0f}s  "
              f"|  {self.n_windows} windows × {self.window_dur:.0f}s each")
        print(f"  Confidence threshold: {self.history.min_confidence:.2f}  "
              f"|  Cooldown: {self.history.cooldown_s:.0f}s")
        print()

    def _print_window_banner(self, idx: int, t_start: float):
        t_end = t_start + self.window_dur
        print()
        print("─" * 66)
        print(f"  WINDOW {idx+1}/{self.n_windows}  "
              f"|  t={t_start:.0f}s–{t_end:.0f}s  "
              f"|  {datetime.now().strftime('%H:%M:%S')}")
        print("─" * 66)

    def _print_window_result(
        self, idx, analysis, effective_intent, reason, did_change, capture_result
    ):
        # Capture stats
        if capture_result:
            n  = capture_result.get("n_frames", "?")
            fp = capture_result.get("fps_achieved", 0.0)
            au = "mic" if capture_result.get("has_real_audio") else "silent"
            print(f"\n  Capture:  {n} frames @ {fp:.1f} fps  |  Audio: {au}")

        # Analysis output
        if analysis is None:
            print("\n  [ANALYSIS FAILED — using fallback]")
        else:
            v     = analysis.get("valence", 0.0)
            a     = analysis.get("arousal", 0.0)
            v_dir = analysis.get("valence_direction", "?")
            a_dir = analysis.get("arousal_direction", "?")
            v_vol = analysis.get("valence_volatility", 0.0)
            a_vol = analysis.get("arousal_volatility", 0.0)
            label = analysis.get("va_state_label", "?")
            conf  = analysis.get("state_confidence", 0.0)
            vol   = analysis.get("volatility", max(v_vol, a_vol))
            ra    = analysis.get("reaction_action")
            mode  = ra.pose_mode if ra else "N/A"

            print(f"\n  VA Baseline:")
            print(f"    Valence:  {v:+.3f}  | Trend: {v_dir:<12} | Volatility: {v_vol:.3f}")
            print(f"    Arousal:  {a:+.3f}  | Trend: {a_dir:<12} | Volatility: {a_vol:.3f}")
            print(f"\n  State:")
            print(f"    Label:      {label}")
            print(f"    Confidence: {conf:.3f}   Volatility: {vol:.3f}")
            print(f"    Pose mode:  {mode}")

            if ra and self.debug:
                print(f"\n  Reaction detail:")
                print(f"    Speed:    {ra.speed_mult:.2f}x  "
                      f"Distance: {ra.distance_mult:.2f}x  "
                      f"Duration: {ra.duration_s:.1f}s")
                print(f"    Explain:  {ra.explain}")

        # Behavior decision
        prev = self.history.history[-2]["intent"] if len(self.history.history) >= 2 else "—"
        labels = {
            "first":          "first window",
            "low_confidence": f"confidence < {self.history.min_confidence:.2f}",
            "cooldown":       "cooldown active",
            "same":           "no change",
            "changed":        "accepted",
        }
        print()
        if did_change:
            print(f"  ★  BEHAVIOR UPDATE:  {prev}  →  {effective_intent}  "
                  f"({labels.get(reason, reason)})")
        else:
            print(f"  ·  Maintaining:  {effective_intent}  ({labels.get(reason, reason)})")

    def _print_summary(self, elapsed_s: float):
        w = 64
        print()
        print("╔" + "═" * w + "╗")
        print("║" + "  SESSION COMPLETE  ".center(w) + "║")
        print("╚" + "═" * w + "╝")
        print(f"  Total time: {elapsed_s:.1f}s")
        print()
        print("  Behavior timeline:")
        print(f"  {'Win':<5} {'t':>5}  {'VA Label':<26} {'Conf':>5}  {'Intent':<15}  Note")
        print("  " + "─" * 68)
        for r in self.session_results:
            a      = r.get("analysis")
            label  = a.get("va_state_label", "error") if a else "error"
            conf   = a.get("state_confidence", 0.0)   if a else 0.0
            intent = r.get("effective_intent", "N/A")
            note   = "← changed" if r.get("did_change") else ""
            t      = r["window_start_s"]
            idx    = r["window_idx"]
            print(f"  {idx+1:<5} {t:>5.0f}s  {label:<26} {conf:>5.2f}  {intent:<15}  {note}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_intent(analysis: Dict) -> str:
    """Pull the intent string from an analysis dict."""
    action = analysis.get("reaction_action")
    return str(action.intent) if action is not None else "NEUTRAL"


def _error_result(idx: int, t_start: float, error: str) -> Dict:
    return {
        "window_idx": idx, "window_start_s": t_start,
        "analysis": None, "effective_intent": "NEUTRAL",
        "reason": error, "did_change": False,
        "capture": None, "window_elapsed_s": 0.0,
    }
