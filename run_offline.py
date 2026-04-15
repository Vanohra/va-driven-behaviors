#!/usr/bin/env python3
"""
Offline VA Pipeline — analyze a pre-recorded video in sliding windows.

Mirrors the online (live) pipeline exactly, but reads from a pre-recorded
video file instead of a webcam ring buffer.  Every --window seconds of video
is treated as one analysis window and processed with the same stack:

  Feature extraction → VA inference → Robust preprocessing → Baseline →
  Trends → Volatility → State classification → Reaction mapping →
  Stability gating (confidence threshold + cooldown)

Output format — BEHAVIOR DISPATCH blocks and a session summary table — is
identical to run_online.py so results from both modes are directly comparable.

Pipeline stages (identical to online):
  1. Feature extraction  — JointCAM audio-visual model
  2. VA inference        — per-frame valence + arousal
  3. Robust preprocessing — winsorize → Hampel filter → rolling-median smoothing
  4. Session baseline    — trimmed-mean estimator
  5. Trend analysis      — Theil-Sen slope + start/end median delta
  6. Volatility          — MAD of preprocessed series
  7. State classification — percentile-based (8 states)
  8. Reaction mapping    — 5-intent policy → ReactionAction
  9. Stability gating    — confidence threshold + cooldown (ReactionHistory)

Usage:
    python run_offline.py path/to/video.mp4
    python run_offline.py path/to/video.mp4 --debug
    python run_offline.py path/to/video.mp4 --window 5
    python run_offline.py path/to/video.mp4 --sparse
    python run_offline.py path/to/video.mp4 --model models/jointcam_finetuned_v4.pt
    python run_offline.py path/to/video.mp4 --device cuda
    python run_offline.py path/to/video.mp4 --min-confidence 0.6 --cooldown 3
"""

import sys
import os
import argparse
import tempfile
import shutil
import time
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline import load_calibration
from online.window_analyzer import WindowAnalyzer
from online.online_session import ReactionHistory, _extract_intent

# ── Optional robot adapter (imported lazily so missing pyserial doesn't crash) ─
def _load_bittle_adapter(port: str, mock: bool):
    """Return a BittleXAdapter, or None if the import fails unexpectedly."""
    try:
        from robot.bittle_adapter import BittleXAdapter
        return BittleXAdapter(port=port, mock=mock)
    except Exception as exc:
        print(f"[BITTLE] Could not initialise adapter: {exc}")
        return None

try:
    from test_emotions import load_model
except ImportError as e:
    print(f"ERROR: Could not import test_emotions: {e}")
    print("  Place test_emotions.py in the project root directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline VA pipeline — analyze a pre-recorded video in windowed mode"
    )
    p.add_argument("video_path", help="Path to the video file")
    p.add_argument(
        "--window", type=float, default=3.0, metavar="S",
        help="Analysis window size in seconds  (default: 3)"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Print verbose debug info at each pipeline stage"
    )
    p.add_argument(
        "--model", default=None, metavar="PATH",
        help="Path to model .pt file  (default: models/jointcam_finetuned_v4.pt)"
    )
    p.add_argument(
        "--calibration", default=None, metavar="PATH",
        help="Path to calibration.json  (default: config/calibration.json)"
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Torch device  (default: cpu)"
    )
    p.add_argument(
        "--audio", default=None, metavar="PATH",
        help="Separate audio file  (default: extracted from video)"
    )
    p.add_argument(
        "--sparse", action="store_true",
        help="Sparse sampling mode: process only 2 frames per window (CPU-efficient)"
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.55, metavar="C",
        help="Minimum state_confidence to accept a new intent  (default: 0.55)"
    )
    p.add_argument(
        "--cooldown", type=float, default=0.5, metavar="S",
        help="Min seconds between behavior changes  (default: 0.5)"
    )
    # ── Bittle hardware ───────────────────────────────────────────────────────
    p.add_argument(
        "--robot-port", default=None, metavar="PORT",
        help="Serial port for Petoi Bittle X  (e.g. COM7, /dev/ttyUSB0). "
             "Omit to run without hardware."
    )
    p.add_argument(
        "--mock-robot", action="store_true",
        help="Enable mock robot mode: prints commands instead of sending to hardware. "
             "Useful for testing the integration without the physical robot connected."
    )
    return p.parse_args()


def find_model(args_model: str, root: Path) -> Path:
    """Resolve model path, trying the preferred model first."""
    if args_model:
        p = Path(args_model)
        if p.exists():
            return p
        print(f"Warning: specified model not found: {p}")
    for name in ("jointcam_finetuned_v4.pt", "jointcam_model.pt"):
        p = root / "models" / name
        if p.exists():
            return p
    print(
        "ERROR: No model file found.\n"
        f"  Place jointcam_finetuned_v4.pt (or jointcam_model.pt) in:  {root / 'models'}\n"
        "  Or specify with:  --model path/to/model.pt"
    )
    sys.exit(1)


# ── Output helpers (matching online pipeline format exactly) ──────────────────

def _print_header(
    video_path: Path,
    window_dur: float,
    n_windows: int,
    model_name: str,
    device: str,
    min_confidence: float,
    cooldown_s: float,
) -> None:
    w = 64
    print()
    print("╔" + "═" * w + "╗")
    print("║" + "  VA OFFLINE PIPELINE  ".center(w) + "║")
    print("║" + "  (Windowed — same analysis loop as online)  ".center(w) + "║")
    print("╚" + "═" * w + "╝")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Video:       {video_path.name}")
    print(f"  Model:       {model_name}")
    print(f"  Device:      {device}")
    print(f"  Windows:     {n_windows} × {window_dur:.0f}s each")
    print(f"  Confidence:  {min_confidence:.2f}  |  Cooldown: {cooldown_s:.1f}s")
    print()


def _print_window_banner(
    window_idx: int, n_windows: int, start_s: float, end_s: float
) -> None:
    print()
    print("─" * 66)
    print(
        f"  WINDOW {window_idx+1}/{n_windows}  "
        f"|  t={start_s:.1f}s–{end_s:.1f}s  "
        f"|  {datetime.now().strftime('%H:%M:%S')}"
    )
    print("─" * 66)


def _dispatch_behavior(
    window_idx: int,
    start_s: float,
    effective_intent: str,
    did_change: bool,
    analysis: Optional[Dict],
) -> None:
    """Print BEHAVIOR DISPATCH block — identical layout to online pipeline."""
    ra    = analysis.get("reaction_action")  if analysis else None
    label = analysis.get("va_state_label",  "unknown") if analysis else "unknown"
    conf  = analysis.get("state_confidence", 0.0)      if analysis else 0.0
    v     = analysis.get("valence",          0.0)      if analysis else 0.0
    a     = analysis.get("arousal",          0.0)      if analysis else 0.0

    change_tag = "CHANGED" if did_change else "maintained"
    bar = "─" * 52

    print(f"\n  ┌─ BEHAVIOR DISPATCH  "
          f"[window {window_idx+1}  t≈{start_s:.0f}s] {bar}┐")
    print(f"  │  Intent :  {effective_intent:<12}  ({change_tag})")
    if ra:
        print(f"  │  Pose   :  {ra.pose_mode:<14}  "
              f"speed={ra.speed_mult:.2f}×  dist={ra.distance_mult:.2f}×  "
              f"dur={ra.duration_s:.1f}s")
    print(f"  │  State  :  {label}  (conf: {conf:.3f})")
    print(f"  │  VA     :  valence={v:+.3f}   arousal={a:+.3f}")
    print(f"  └{bar}──────────────────────┘")


def _print_window_detail(
    analysis: Optional[Dict],
    effective_intent: str,
    reason: str,
    did_change: bool,
    n_frames: int,
    analysis_elapsed: float,
    history: ReactionHistory,
    debug: bool,
) -> None:
    """Print per-window detail — matches online _print_window_result layout."""
    print(f"\n  Frames: {n_frames}  |  Analysis time: {analysis_elapsed:.2f}s")

    if analysis is None:
        print("\n  [ANALYSIS FAILED — using fallback]")
    else:
        v     = analysis.get("valence",           0.0)
        a     = analysis.get("arousal",            0.0)
        v_dir = analysis.get("valence_direction",  "?")
        a_dir = analysis.get("arousal_direction",  "?")
        v_vol = analysis.get("valence_volatility", 0.0)
        a_vol = analysis.get("arousal_volatility", 0.0)
        label = analysis.get("va_state_label",     "?")
        conf  = analysis.get("state_confidence",   0.0)
        vol   = analysis.get("volatility",         max(v_vol, a_vol))
        ra    = analysis.get("reaction_action")
        mode  = ra.pose_mode if ra else "N/A"

        print(f"\n  VA Baseline:")
        print(f"    Valence:  {v:+.3f}  | Trend: {v_dir:<12} | Volatility: {v_vol:.3f}")
        print(f"    Arousal:  {a:+.3f}  | Trend: {a_dir:<12} | Volatility: {a_vol:.3f}")
        print(f"\n  State:")
        print(f"    Label:      {label}")
        print(f"    Confidence: {conf:.3f}   Volatility: {vol:.3f}")
        print(f"    Pose mode:  {mode}")

        if ra and debug:
            print(f"\n  Reaction detail:")
            print(f"    Speed:    {ra.speed_mult:.2f}x  "
                  f"Distance: {ra.distance_mult:.2f}x  "
                  f"Duration: {ra.duration_s:.1f}s")
            print(f"    Explain:  {ra.explain}")

    prev = history.history[-2]["intent"] if len(history.history) >= 2 else "—"
    labels = {
        "first":          "first window",
        "low_confidence": f"confidence < {history.min_confidence:.2f}",
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


def _print_summary(
    session_results: List[Dict], elapsed_s: float, window_dur: float
) -> None:
    """Print end-of-session summary — identical to online StreamingSession._print_summary."""
    w = 64
    print()
    print("╔" + "═" * w + "╗")
    print("║" + "  SESSION COMPLETE  ".center(w) + "║")
    print("╚" + "═" * w + "╝")
    print(f"  Total time: {elapsed_s:.1f}s")
    print()
    print("  Behavior timeline:")
    print(
        f"  {'Win':<5} {'t':>6}  {'VA Label':<26} "
        f"{'Conf':>5}  {'Analysis':>9}  {'Intent':<15}  Note"
    )
    print("  " + "─" * 80)
    for r in session_results:
        a      = r.get("analysis")
        label  = a.get("va_state_label",   "error") if a else "error"
        conf   = a.get("state_confidence",  0.0)    if a else 0.0
        intent = r.get("effective_intent",  "N/A")
        note   = "← changed" if r.get("did_change") else ""
        t      = r["window_start_s"]
        idx    = r["window_idx"]
        timing = r.get("analysis_elapsed_s", 0.0)
        flag   = " !" if timing > window_dur else "  "
        print(
            f"  {idx+1:<5} {t:>5.1f}s  {label:<26} "
            f"{conf:>5.2f}  {timing:>7.2f}s{flag}  {intent:<15}  {note}"
        )
    print()
    print("  Note: '!' in the Analysis column means that window's analysis")
    print(f"        exceeded the {window_dur:.0f}s window duration.")
    print()


# ── Core windowed session (mirrors StreamingSession, reads from file) ─────────

def run_offline_session(
    video_path: Path,
    analyzer: WindowAnalyzer,
    window_dur: float,
    audio_path: Optional[str],
    sparse: bool,
    min_confidence: float,
    cooldown_s: float,
    debug: bool,
    robot=None,           # BittleXAdapter instance, or None for terminal-only mode
) -> List[Dict]:
    """
    Process a video file in sequential windows using the same analysis loop
    and stability gating as the online streaming session.

    Returns a list of per-window result dicts in the same schema as
    StreamingSession.session_results.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / fps
    n_windows    = max(1, int(duration_s / window_dur))
    cap.release()

    history = ReactionHistory(min_confidence=min_confidence, cooldown_s=cooldown_s)
    session_results: List[Dict] = []
    num_samples = 2 if sparse else None
    last_v: Optional[float] = None
    last_a: Optional[float] = None

    tmp_dir = Path(tempfile.mkdtemp(prefix="va_offline_"))
    session_start = time.time()

    try:
        for window_idx in range(n_windows):
            start_s = window_idx * window_dur
            end_s   = min(start_s + window_dur, duration_s)
            if (end_s - start_s) < 1.0:
                break  # Skip final fragment too short to analyze

            _print_window_banner(window_idx, n_windows, start_s, end_s)

            # ── 1. Extract video segment ──────────────────────────────────────
            seg_path = tmp_dir / f"win_{window_idx}.mp4"
            cap    = cv2.VideoCapture(str(video_path))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(str(seg_path), fourcc, fps, (frame_w, frame_h))
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
            n_frames_written = 0
            target = int((end_s - start_s) * fps)
            for _ in range(target):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                n_frames_written += 1
            out.release()
            cap.release()

            # ── 2. Run VA analysis ────────────────────────────────────────────
            print(f"  Running VA analysis pipeline...", flush=True)
            t_start  = time.time()
            analysis = analyzer.analyze_window(
                video_path=str(seg_path),
                audio_path=audio_path,
                num_samples=num_samples,
                last_v=last_v,
                last_a=last_a,
            )
            analysis_elapsed = time.time() - t_start

            # ── 3. Stability gating (identical to online ReactionHistory) ─────
            now = time.time()
            if analysis is not None:
                proposed = _extract_intent(analysis)
                conf     = analysis.get("state_confidence", 0.0)
                label    = analysis.get("va_state_label", "unknown")
                last_v   = analysis.get("valence")
                last_a   = analysis.get("arousal")
            else:
                proposed = history.fallback_intent
                conf     = 0.0
                label    = "unknown"

            effective, reason, did_change = history.evaluate(proposed, conf, label, now)

            # ── 4. Dispatch + per-window detail ───────────────────────────────
            _dispatch_behavior(window_idx, start_s, effective, did_change, analysis)

            # ── 4a. Send reaction to physical Bittle (if adapter is active) ───
            if robot is not None and analysis is not None:
                ra = analysis.get("reaction_action")
                if ra is not None:
                    ra.intent = effective  # respect stability gate — not raw model intent
                    robot.apply_reaction(ra)

            _print_window_detail(
                analysis, effective, reason, did_change,
                n_frames_written, analysis_elapsed, history, debug,
            )

            # ── 5. Store result (same schema as StreamingSession) ─────────────
            session_results.append({
                "window_idx":         window_idx,
                "window_start_s":     start_s,
                "analysis":           analysis,
                "effective_intent":   effective,
                "reason":             reason,
                "did_change":         did_change,
                "n_frames":           n_frames_written,
                "analysis_elapsed_s": analysis_elapsed,
            })

            # ── 6. Cleanup segment ────────────────────────────────────────────
            try:
                os.remove(str(seg_path))
            except OSError:
                pass

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            print(f"\nNote: Temp directory {tmp_dir} remains. You may delete it manually.")

    _print_summary(session_results, time.time() - session_start, window_dur)
    return session_results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    root = Path(__file__).parent

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    model_path = find_model(args.model, root)
    calib_path = (
        Path(args.calibration) if args.calibration
        else root / "config" / "calibration.json"
    )

    calibration = None
    if calib_path.exists():
        calibration = load_calibration(str(calib_path))
        if calibration:
            print(f"Calibration loaded: {calib_path.name}")
        else:
            print("Warning: calibration.json invalid — using fallback thresholds.")
    else:
        print(
            f"Warning: calibration.json not found at {calib_path} "
            "— using fallback thresholds."
        )

    # Read video metadata for header
    cap        = cv2.VideoCapture(str(video_path))
    fps_raw    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = n_frames / fps_raw
    n_windows  = max(1, int(duration_s / args.window))

    print(f"Loading model: {model_path.name} ...", end=" ", flush=True)
    model = load_model(str(model_path), args.device)
    print("done.")

    analyzer = WindowAnalyzer(
        model=model,
        device=args.device,
        calibration=calibration,
        debug=args.debug,
    )

    _print_header(
        video_path, args.window, n_windows,
        model_path.name, args.device,
        args.min_confidence, args.cooldown,
    )

    # ── Optional Bittle hardware setup ────────────────────────────────────────
    robot = None
    if args.robot_port or args.mock_robot:
        port  = args.robot_port or "COM7"
        robot = _load_bittle_adapter(port=port, mock=args.mock_robot)
        if robot is not None:
            mode = "mock" if robot.mock else f"hardware on {port}"
            print(f"  Robot:       Petoi Bittle X ({mode})")

    try:
        run_offline_session(
            video_path=video_path,
            analyzer=analyzer,
            window_dur=args.window,
            audio_path=args.audio,
            sparse=args.sparse,
            min_confidence=args.min_confidence,
            cooldown_s=args.cooldown,
            debug=args.debug,
            robot=robot,
        )
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
