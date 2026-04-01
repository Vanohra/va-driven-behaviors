#!/usr/bin/env python3
"""
Generate Calibration JSON from Batch of Videos

This script processes a batch of videos to compute calibration statistics
(valence and arousal min/max/mean/std/percentiles) and saves them to calibration.json.

This calibration file is then used by the emotion analysis system for
scale-aware threshold calculations.

Usage:
    python generate_calibration.py <video_dir> [--model_path <model.pt>] [--device <cpu|cuda>] [--output <calibration.json>]
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    from test_emotions import (
        load_model,
        extract_video_features,
        extract_audio_features,
        align_features,
        predict_emotions
    )
except ImportError as e:
    print(f"Error importing test_emotions: {e}")
    print("Place test_emotions.py in the project root directory.")
    sys.exit(1)

from .emotion_analyzer import compute_calibration_stats, print_calibration_stats, preprocess_series, compute_robust_baseline


def process_video_for_calibration(video_path, model, device='cpu', verbose=False):
    """
    Process a single video and return valence/arousal statistics.
    
    Args:
        video_path: Path to video file
        model: Loaded JointCAM model
        device: 'cpu' or 'cuda'
        verbose: Whether to print progress
    
    Returns:
        Dictionary with 'valence' and 'arousal' keys, or None if error
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        if verbose:
            print(f"  Error: File not found: {video_path}")
        return None
    
    try:
        if verbose:
            print(f"  Processing: {video_path.name}")
        
        # Extract features
        video_features, fps = extract_video_features(video_path, device)
        audio_features = extract_audio_features(video_path)
        
        # Align features
        video_aligned, audio_aligned = align_features(video_features, audio_features)
        
        # Run inference - returns per-frame arrays
        valence_series, arousal_series = predict_emotions(
            model, video_aligned, audio_aligned, device
        )
        
        # Ensure 1D arrays
        if valence_series.ndim > 1:
            valence_series = valence_series.flatten()
        if arousal_series.ndim > 1:
            arousal_series = arousal_series.flatten()
        
        # Robust preprocessing configuration (same as in emotion_analyzer)
        preprocess_config = {
            'winsor_limits': (0.01, 0.01),
            'hampel_window': 7,
            'hampel_n_sigmas': 3.0,
            'smooth_window': 5,
            'smooth_method': 'rolling_median'
        }
        
        # Preprocess both series
        valence_preprocessed, valence_meta = preprocess_series(valence_series, preprocess_config)
        arousal_preprocessed, arousal_meta = preprocess_series(arousal_series, preprocess_config)
        
        # Compute robust session baseline (trimmed mean)
        valence_baseline = compute_robust_baseline(
            valence_series,
            preprocessed_x=valence_preprocessed,
            trim_ratio=0.10,
            outlier_rate=valence_meta['outlier_rate']
        )
        arousal_baseline = compute_robust_baseline(
            arousal_series,
            preprocessed_x=arousal_preprocessed,
            trim_ratio=0.10,
            outlier_rate=arousal_meta['outlier_rate']
        )
        
        # Use robust baseline value (trimmed mean) for calibration
        mean_valence = valence_baseline['value']
        mean_arousal = arousal_baseline['value']
        
        if verbose:
            print(f"    Valence: {mean_valence:.4f}, Arousal: {mean_arousal:.4f}")
        
        return {
            'valence': mean_valence,
            'arousal': mean_arousal
        }
        
    except Exception as e:
        if verbose:
            print(f"  Error processing {video_path.name}: {e}")
        return None


def generate_calibration(video_dir, model_path='jointcam_model.pt', device='cpu',
                        output_file='calibration.json', recursive=True, verbose=True):
    """
    Generate calibration.json from a batch of videos.
    
    Args:
        video_dir: Directory containing video files
        model_path: Path to model checkpoint
        device: 'cpu' or 'cuda'
        output_file: Output calibration.json file path
        recursive: Whether to search subdirectories recursively
        verbose: Whether to print progress
    
    Returns:
        Path to created calibration.json file, or None if error
    """
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        print(f"Error: Directory does not exist: {video_dir}")
        sys.exit(1)
    
    if not video_dir.is_dir():
        print(f"Error: Path is not a directory: {video_dir}")
        sys.exit(1)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    
    if recursive:
        for ext in video_extensions:
            video_files.extend(video_dir.rglob(f'*{ext}'))
    else:
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
    
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files")
    print()
    
    # Load model once
    if verbose:
        print("Loading JointCAM model...")
    
    if not Path(model_path).is_absolute() and Path(model_path).exists():
        model_path = str(Path(model_path).absolute())
    
    model = load_model(model_path, device)
    if verbose:
        print(f"Model loaded from: {model_path}")
        print()
    
    # Process all videos to collect VA statistics
    print("=" * 60)
    print("CALIBRATION PASS: Processing videos to collect VA statistics")
    print("=" * 60)
    print()
    
    calibration_results = []
    errors = []
    
    for i, video_file in enumerate(video_files, 1):
        if verbose:
            print(f"[{i}/{len(video_files)}] {video_file.relative_to(video_dir)}")
        
        result = process_video_for_calibration(
            video_file, model, device, verbose=verbose
        )
        
        if result:
            calibration_results.append(result)
        else:
            errors.append(str(video_file))
        
        if verbose:
            print()
    
    if not calibration_results:
        print("Error: No videos processed successfully. Cannot generate calibration.")
        sys.exit(1)
    
    print(f"Successfully processed: {len(calibration_results)}/{len(video_files)} videos")
    if errors:
        print(f"Errors: {len(errors)} videos")
        if verbose:
            print("Failed videos:")
            for err in errors[:10]:  # Show first 10
                print(f"  {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
    print()
    
    # Compute calibration statistics
    print("Computing calibration statistics...")
    calibration = compute_calibration_stats(calibration_results)
    
    if not calibration:
        print("Error: Could not compute calibration statistics.")
        sys.exit(1)
    
    # Print calibration stats
    print_calibration_stats(calibration)
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"\nCalibration saved to: {output_path}")
    print(f"Absolute path: {output_path.absolute()}")
    print()
    print("=" * 60)
    print("CALIBRATION GENERATION COMPLETE")
    print("=" * 60)
    print()
    print(f"You can now use this calibration file with:")
    print(f"  python simulation/emotional_controller.py <video> --calibration {output_path}")
    print()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration.json from batch of videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_calibration.py ravdess_video_speech/
    python generate_calibration.py ravdess_video_speech/ --output calibration.json
    python generate_calibration.py ravdess_video_speech/ --device cuda --model_path custom_model.pt
        """
    )
    parser.add_argument("video_dir", help="Directory containing video files")
    parser.add_argument(
        "--model_path",
        default="jointcam_model.pt",
        help="Path to JointCAM model checkpoint (default: jointcam_model.pt)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)"
    )
    parser.add_argument(
        "--output",
        default="calibration.json",
        help="Output calibration.json file path (default: calibration.json)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Calibration Generation from Batch Videos")
    print("=" * 60)
    print()
    print(f"Video directory: {args.video_dir}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print()
    
    generate_calibration(
        args.video_dir,
        model_path=args.model_path,
        device=args.device,
        output_file=args.output,
        recursive=not args.no_recursive,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
