"""
Evaluation 3 — Raw VA vs. preprocessed VA signal quality.

Target:  pipeline/emotion_analyzer.py — preprocess_series (Winsorization,
         Hampel filter, rolling-median smoothing).
         The robot is NOT involved.

Purpose:
    Show that preprocessing produces a cleaner VA signal than raw frame-by-frame
    output, justifying its use before intent selection.

Method:
    For each video in samples/:
        1. Run feature extraction + emotion inference to get raw valence and
           arousal series (identical to what WindowAnalyzer does internally).
        2. Apply the same preprocess_series() call that analyze_emotion_stream
           uses internally, capturing the preprocessed series and metadata.
        3. Compute per-video, per-channel metrics on both raw and processed:
               mad                  — median absolute deviation
               iqr                  — interquartile range
               frame_delta_variance — variance of np.diff(series) (jitter proxy)
               outlier_rate         — (n_winsorized + n_hampel_replaced) / (2 × n_frames)
                                      averaged over valence + arousal; 0.0 for raw
               n_frames             — length of raw series
        4. Save the first video's raw/processed series to
           eval_3_sample_series.npz for notebook time-series visualisation.

NOTE: This eval re-runs the vision model (~25 s per video, ~12–15 min total).
      Use --sample N to test on a subset first.

Usage:
    python evaluation/eval_3_va_preprocessing/run_eval.py --sample 1   # smoke-test
    python evaluation/eval_3_va_preprocessing/run_eval.py               # all 30
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
EVAL_DIR     = Path(__file__).resolve().parent.parent   # evaluation/
ROOT         = EVAL_DIR.parent                           # project root
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
SERIES_PATH  = Path(__file__).resolve().parent / "eval_3_sample_series.npz"
SAMPLES_DIR  = ROOT / "samples"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from pipeline.emotion_analyzer import preprocess_series
from pipeline.robust_stats import mad, iqr
from eval_utils import save_results

# Preprocessing config — mirrors the config inside analyze_emotion_stream
PREPROCESS_CONFIG = {
    "winsor_limits":   (0.01, 0.01),
    "hampel_window":   7,
    "hampel_n_sigmas": 3.0,
    "smooth_window":   5,
    "smooth_method":   "rolling_median",
}


# ── Signal metrics ────────────────────────────────────────────────────────────

def compute_signal_metrics(series: np.ndarray) -> dict:
    """Compute MAD, IQR, frame_delta_variance on a 1D float series."""
    x = series[~np.isnan(series)]
    if len(x) < 2:
        return {"mad": 0.0, "iqr": 0.0, "frame_delta_variance": 0.0}
    return {
        "mad":                  float(mad(x)),
        "iqr":                  float(iqr(x)),
        "frame_delta_variance": float(np.var(np.diff(x))),
    }


# ── Per-video processing ──────────────────────────────────────────────────────

def process_video(video_path: Path, model, device: str) -> list[dict] | None:
    """
    Extract raw + preprocessed VA series from one video.

    Returns a list of 4 result rows (raw/processed × valence/arousal),
    or None on failure.
    """
    try:
        from test_emotions import (
            extract_video_features,
            extract_audio_features,
            align_features,
            predict_emotions,
        )
    except ImportError as e:
        print(f"[EVAL 3] Cannot import test_emotions: {e}")
        return None

    try:
        video_features, _fps = extract_video_features(
            str(video_path), device, num_samples=None
        )
        audio_features = extract_audio_features(str(video_path))
        video_aligned, audio_aligned = align_features(video_features, audio_features)
        valence_raw, arousal_raw = predict_emotions(
            model, video_aligned, audio_aligned, device
        )
        valence_raw = np.asarray(valence_raw, dtype=np.float64).flatten()
        arousal_raw = np.asarray(arousal_raw, dtype=np.float64).flatten()
    except Exception as e:
        print(f"\n[EVAL 3] Feature extraction/inference failed for "
              f"{video_path.name}: {e}")
        return None

    n_frames = len(valence_raw)
    if n_frames < 2:
        print(f"\n[EVAL 3] Too few frames ({n_frames}) in {video_path.name} — skipping.")
        return None

    # Apply preprocessing (replicates analyze_emotion_stream internals)
    valence_proc, v_meta = preprocess_series(valence_raw, PREPROCESS_CONFIG)
    arousal_proc, a_meta = preprocess_series(arousal_raw, PREPROCESS_CONFIG)

    # Average outlier rate across both channels (characterises the full window)
    n_outliers = (
        v_meta["n_winsorized"] + v_meta["n_hampel_replaced"] +
        a_meta["n_winsorized"] + a_meta["n_hampel_replaced"]
    )
    outlier_rate_processed = n_outliers / (2 * n_frames) if n_frames > 0 else 0.0

    stem = video_path.stem
    rows: list[dict] = []

    for channel, raw_s, proc_s in [
        ("valence", valence_raw, valence_proc),
        ("arousal", arousal_raw, arousal_proc),
    ]:
        raw_m  = compute_signal_metrics(raw_s)
        proc_m = compute_signal_metrics(proc_s)

        rows.append({
            "video":               stem,
            "condition":           "raw",
            "channel":             channel,
            "mad":                 round(raw_m["mad"], 6),
            "iqr":                 round(raw_m["iqr"], 6),
            "frame_delta_variance": round(raw_m["frame_delta_variance"], 6),
            "outlier_rate":        0.0,
            "n_frames":            n_frames,
        })
        rows.append({
            "video":               stem,
            "condition":           "processed",
            "channel":             channel,
            "mad":                 round(proc_m["mad"], 6),
            "iqr":                 round(proc_m["iqr"], 6),
            "frame_delta_variance": round(proc_m["frame_delta_variance"], 6),
            "outlier_rate":        round(outlier_rate_processed, 6),
            "n_frames":            n_frames,
        })

    return rows, (valence_raw, arousal_raw, valence_proc, arousal_proc)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(sample_n: int) -> None:
    video_files = sorted(SAMPLES_DIR.glob("*.mp4"))
    if not video_files:
        print(f"[EVAL 3] No .mp4 files found in {SAMPLES_DIR}")
        return

    if sample_n < len(video_files):
        video_files = video_files[:sample_n]
        print(f"[EVAL 3] --sample {sample_n}: processing first {sample_n} of "
              f"{len(sorted(SAMPLES_DIR.glob('*.mp4')))} videos")

    # Load model once (same lookup as run_offline.py)
    try:
        from run_offline import find_model
        from test_emotions import load_model
    except ImportError as e:
        print(f"[EVAL 3] Cannot import pipeline modules: {e}")
        return

    model_path = find_model(None, ROOT)
    device = "cpu"
    print(f"[EVAL 3] Loading model: {model_path.name} ...", end=" ", flush=True)
    model = load_model(str(model_path), device)
    print("done.")

    all_rows: list[dict] = []
    sample_series_saved = False

    for i, video_path in enumerate(video_files):
        print(f"  [{i+1}/{len(video_files)}] {video_path.name}", end=" ... ", flush=True)
        t0 = time.time()
        result = process_video(video_path, model, device)
        elapsed = time.time() - t0

        if result is None:
            print("FAILED")
            continue

        rows, series_tuple = result
        all_rows.extend(rows)
        print(f"done ({elapsed:.1f}s, {rows[0]['n_frames']} frames)")

        # Save first successful video's series for notebook visualisation
        if not sample_series_saved:
            valence_raw, arousal_raw, valence_proc, arousal_proc = series_tuple
            np.savez(
                SERIES_PATH,
                video_name=np.array(video_path.stem),
                valence_raw=valence_raw,
                arousal_raw=arousal_raw,
                valence_processed=valence_proc,
                arousal_processed=arousal_proc,
            )
            print(f"  [sample series saved → {SERIES_PATH.name}]")
            sample_series_saved = True

    if not all_rows:
        print("[EVAL 3] No results — nothing saved.")
        return

    save_results(all_rows, RESULTS_PATH)
    print(f"\n[EVAL 3] Saved {len(all_rows)} rows to {RESULTS_PATH}")
    print(f"  ({len(video_files)} videos × 2 conditions × 2 channels = "
          f"{len(video_files) * 4} expected rows)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Eval 3: Raw vs. preprocessed VA signal quality"
    )
    p.add_argument(
        "--sample", type=int, default=30, metavar="N",
        help="Process only the first N videos from samples/ (default: 30 = all)"
    )
    args = p.parse_args()
    run(args.sample)


if __name__ == "__main__":
    main()
