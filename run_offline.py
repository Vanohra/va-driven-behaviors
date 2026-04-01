#!/usr/bin/env python3
"""
Offline VA Pipeline — analyze a pre-recorded video file.

Runs the full valence-arousal pipeline on a video file and prints the
analysis results to the terminal.  Does NOT require PyBullet or any
simulation software.

Pipeline stages (all preserved from the original research paper):
  1. Feature extraction  — JointCAM audio-visual model
  2. VA inference        — per-frame valence + arousal
  3. Robust preprocessing — winsorize → Hampel filter → EMA smoothing
  4. Session baseline    — trimmed-mean estimator
  5. Trend analysis      — Theil-Sen slope + start/end median delta
  6. Volatility          — MAD of preprocessed series
  7. State classification — percentile-based (8 states)
  8. Reaction mapping    — 5-intent policy → ReactionAction

Usage:
    python run_offline.py path/to/video.mp4
    python run_offline.py path/to/video.mp4 --debug
    python run_offline.py path/to/video.mp4 --model models/jointcam_finetuned_v4.pt
    python run_offline.py path/to/video.mp4 --device cuda
"""

import sys
import os
import argparse
from pathlib import Path

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline import load_calibration
from online.window_analyzer import WindowAnalyzer

try:
    from test_emotions import load_model
except ImportError as e:
    print(f"ERROR: Could not import test_emotions: {e}")
    print("  Place test_emotions.py in the project root directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline VA pipeline — analyze a pre-recorded video"
    )
    p.add_argument("video_path", help="Path to the video file")
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


def print_result(result: dict) -> None:
    """Print analysis results in a readable format."""
    print()
    print("=" * 62)
    print("  VA ANALYSIS RESULTS")
    print("=" * 62)

    v      = result.get("valence", 0.0)
    a      = result.get("arousal", 0.0)
    v_dir  = result.get("valence_direction", "?")
    a_dir  = result.get("arousal_direction", "?")
    v_vol  = result.get("valence_volatility", 0.0)
    a_vol  = result.get("arousal_volatility", 0.0)
    label  = result.get("va_state_label", "?")
    conf   = result.get("state_confidence", 0.0)
    vol    = result.get("volatility", max(v_vol, a_vol))

    print(f"\n  VA Baseline:")
    print(f"    Valence:  {v:+.4f}  | Trend: {v_dir:<12} | Volatility: {v_vol:.4f}")
    print(f"    Arousal:  {a:+.4f}  | Trend: {a_dir:<12} | Volatility: {a_vol:.4f}")

    print(f"\n  State Classification:")
    print(f"    VA Label:    {label}")
    print(f"    Confidence:  {conf:.4f}")
    print(f"    Volatility:  {vol:.4f}")

    ra = result.get("reaction_action")
    if ra:
        print(f"\n  Reaction Mapping:")
        print(f"    Intent:      {ra.intent}")
        print(f"    Pose mode:   {ra.pose_mode}")
        print(f"    Speed mult:  {ra.speed_mult:.2f}x")
        print(f"    Dist. mult:  {ra.distance_mult:.2f}x")
        print(f"    Duration:    {ra.duration_s:.1f}s")
        print(f"    Explanation: {ra.explain}")

    print()


def main() -> None:
    args = parse_args()
    root = Path(__file__).parent

    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Resolve model and calibration paths
    model_path = find_model(args.model, root)
    calib_path = Path(args.calibration) if args.calibration else root / "config" / "calibration.json"

    calibration = None
    if calib_path.exists():
        calibration = load_calibration(str(calib_path))
        if calibration:
            print(f"Calibration loaded: {calib_path.name}")
        else:
            print("Warning: calibration.json invalid — using fallback thresholds.")
    else:
        print(f"Warning: calibration.json not found at {calib_path} — using fallback thresholds.")

    # Banner
    print()
    print("=" * 62)
    print("  OFFLINE VA PIPELINE")
    print("=" * 62)
    print(f"  Video:    {video_path.name}")
    print(f"  Model:    {model_path.name}")
    print(f"  Device:   {args.device}")
    print(f"  Audio:    {args.audio or '(extracted from video)'}")
    print()

    # Load model (once)
    print("Loading model...", end=" ", flush=True)
    model = load_model(str(model_path), args.device)
    print("done.")

    # Run analysis
    analyzer = WindowAnalyzer(
        model=model,
        device=args.device,
        calibration=calibration,
        debug=args.debug,
    )

    print("Running VA analysis...", flush=True)
    result = analyzer.analyze_window(
        video_path=str(video_path),
        audio_path=args.audio,
    )

    if result is None:
        print("\nERROR: Analysis failed. Run with --debug for details.")
        sys.exit(1)

    print_result(result)


if __name__ == "__main__":
    main()
