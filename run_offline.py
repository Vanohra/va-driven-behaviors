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
import tempfile
import shutil
import cv2
import numpy as np
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
    p.add_argument(
        "--sparse", action="store_true",
        help="Sparse sampling mode: process only 2 frames (CPU-efficient)"
    )
    p.add_argument(
        "--windowed", action="store_true",
        help="Run analysis in 3s windows to evaluate momentum logic"
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


def run_windowed_session(video_path: Path, 
                         analyzer: WindowAnalyzer, 
                         sparse: bool) -> None:
    """Analyze a video in sequence of 3.0s windows (Standard for SIGDIAL)."""
    step_s = 3.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"  Windowed Mode: {step_s}s steps | Total duration: {duration:.1f}s")
    
    last_v, last_a = None, None
    window_count = 0
    
    # Process the video in chunks
    tmp_dir = Path(tempfile.mkdtemp(prefix="va_offline_"))
    try:
        start_time = 0.0
        while start_time < duration:
            end_time = min(start_time + step_s, duration)
            if (end_time - start_time) < 1.0: break # Skip final tiny fragment
            
            window_count += 1
            print(f"\n--- Window {window_count} [{start_time:.1f}s - {end_time:.1f}s] ---")
            
            # Extract segment to temp file
            seg_path = tmp_dir / f"win_{window_count}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(seg_path), fourcc, fps, 
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            target_frames = int((end_time - start_time) * fps)
            for _ in range(target_frames):
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            out.release()
            
            # Analyze this specific segment
            num_samples = 2 if sparse else None
            result = analyzer.analyze_window(
                video_path=str(seg_path),
                num_samples=num_samples,
                last_v=last_v,
                last_a=last_a
            )
            
            if result:
                print_result(result)
                # Chain the state for the next window's momentum
                last_v = result.get("valence")
                last_a = result.get("arousal")
            else:
                print("  Analysis failed for this window.")
            
            start_time += step_s
            
    finally:
        cap.release()
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            # On Windows, temp folders can sometimes be locked temporarily
            print(f"\nNote: Temp directory {tmp_dir} remains. You may delete it manually.")


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

    if args.windowed:
        run_windowed_session(video_path, analyzer, args.sparse)
    else:
        print("Running VA analysis...", flush=True)
        num_samples = 2 if args.sparse else None
        
        result = analyzer.analyze_window(
            video_path=str(video_path),
            audio_path=args.audio,
            num_samples=num_samples
        )

        if result is None:
            print("\nERROR: Analysis failed. Run with --debug for details.")
            sys.exit(1)

        print_result(result)


if __name__ == "__main__":
    main()
