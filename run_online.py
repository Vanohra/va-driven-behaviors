#!/usr/bin/env python3
"""
Online VA Pipeline — live streaming interaction session.

Uses a ring buffer + pipelined architecture:
  - The camera runs continuously for the full session (never pauses).
  - Every window_dur seconds a snapshot of the ring buffer is taken and
    submitted to a worker thread for analysis.
  - Analysis runs in parallel with the next capture window.
  - Analysis timing is printed for each window so you can detect overflow.

Session structure (default, 3 × 10s):
  t=0:   camera starts, ring buffer fills
  t=10s: snapshot 1 taken → worker 1 starts analyzing
  t=20s: snapshot 2 taken → worker 2 starts (worker 1 may still be running)
  t=30s: snapshot 3 taken → worker 3 starts; session timer ends
         executor waits for all in-flight workers before printing summary

Usage:
    python run_online.py
    python run_online.py --debug
    python run_online.py --no-audio          # skip microphone
    python run_online.py --camera 1          # second webcam
    python run_online.py --duration 60 --window 15   # 4 × 15s windows
    python run_online.py --min-confidence 0.6 --cooldown 6
"""

import sys
import argparse
from pathlib import Path

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline import load_calibration
from online.window_analyzer    import WindowAnalyzer
from online.streaming_session  import StreamingSession

try:
    from test_emotions import load_model
except ImportError as e:
    print(f"ERROR: Could not import test_emotions: {e}")
    print("  Place test_emotions.py in the project root directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Online VA pipeline — live 30-second interaction session"
    )
    p.add_argument("--debug",    action="store_true",
                   help="Verbose debug output at each pipeline stage")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable microphone capture (video-only mode)")
    p.add_argument("--camera",   type=int,   default=0,    metavar="N",
                   help="OpenCV camera index  (default: 0)")
    p.add_argument("--fps",      type=float, default=15.0, metavar="F",
                   help="Webcam capture frame rate  (default: 15)")
    p.add_argument("--duration", type=float, default=30.0, metavar="S",
                   help="Total session duration in seconds  (default: 30)")
    p.add_argument("--window",   type=float, default=3.0, metavar="S",
                   help="Analysis window size in seconds  (default: 3)")
    p.add_argument("--min-confidence", type=float, default=0.55, metavar="C",
                   help="Minimum state_confidence to accept a new intent  (default: 0.55)")
    p.add_argument("--cooldown", type=float, default=1.5, metavar="S",
                   help="Min seconds between behavior changes  (default: 1.5)")
    p.add_argument("--model",    default=None, metavar="PATH",
                   help="Path to .pt model file  (default: models/jointcam_finetuned_v4.pt)")
    p.add_argument("--device",   default="cpu", choices=["cpu", "cuda"],
                   help="Torch device  (default: cpu)")
    p.add_argument("--no-cleanup", action="store_true",
                   help="Keep temp video/audio files after the session")
    p.add_argument("--sparse", action="store_true",
                   help="Sparse sampling mode: process only 2 frames per window (CPU-efficient)")
    return p.parse_args()


def find_model(args_model, root: Path) -> Path:
    if args_model:
        p = Path(args_model)
        if p.exists():
            return p
    for name in ("jointcam_finetuned_v4.pt", "jointcam_model.pt"):
        p = root / "models" / name
        if p.exists():
            return p
    print(
        "ERROR: No model file found.\n"
        f"  Place jointcam_finetuned_v4.pt in:  {root / 'models'}\n"
        "  Or specify with:  --model path/to/model.pt"
    )
    sys.exit(1)


def main() -> None:
    args = parse_args()
    root = Path(__file__).parent

    model_path = find_model(args.model, root)
    calib_path = root / "config" / "calibration.json"

    # ── Load calibration ──────────────────────────────────────────────────────
    calibration = None
    if calib_path.exists():
        calibration = load_calibration(str(calib_path))
        if calibration:
            print(f"Calibration loaded: {calib_path.name}")
    if not calibration:
        print("Warning: Using fallback calibration thresholds.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path.name} ...", end=" ", flush=True)
    model = load_model(str(model_path), args.device)
    print("done.")

    # ── Banner ────────────────────────────────────────────────────────────────
    n_windows = int(args.duration / args.window)
    print()
    print("=" * 62)
    print("  VA ONLINE PIPELINE — STREAMING SESSION")
    print("  (Ring Buffer + Pipelined Architecture)")
    print("=" * 62)
    print(f"  Duration:  {args.duration:.0f}s  ({n_windows} windows × {args.window:.0f}s)")
    print(f"  Camera:    index {args.camera}  |  FPS: {args.fps}")
    print(f"  Audio:     {'disabled (--no-audio)' if args.no_audio else 'enabled'}")
    print(f"  Device:    {args.device}")
    print(f"  Model:     {model_path.name}")
    print()

    # ── Build pipeline components ─────────────────────────────────────────────
    analyzer = WindowAnalyzer(
        model=model,
        device=args.device,
        calibration=calibration,
        debug=args.debug,
    )

    # ── Run streaming session ─────────────────────────────────────────────────
    num_samples = 2 if args.sparse else None
    
    try:
        session = StreamingSession(
            window_analyzer=analyzer,
            camera_index=args.camera,
            fps=args.fps,
            resolution=(320, 240),
            session_duration_s=args.duration,
            window_duration_s=args.window,
            capture_audio=not args.no_audio,
            min_confidence=args.min_confidence,
            cooldown_s=args.cooldown,
            debug=args.debug,
            cleanup_temp=not args.no_cleanup,
            num_samples=num_samples,
        )
        session.run()

    except KeyboardInterrupt:
        print("\n\n  Session interrupted (Ctrl+C).")
    except Exception as e:
        print(f"\n  FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("  Done.")


if __name__ == "__main__":
    main()
