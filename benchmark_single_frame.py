#!/usr/bin/env python3
"""
Single-frame timing benchmark.

Measures how long ResNet50 feature extraction and JointCAM inference each
take on a single frame, so you can get a feel for the model speed independent
of video I/O and windowing overhead.

Uses the first frame of the first video found in samples/.

Usage:
    python benchmark_single_frame.py
    python benchmark_single_frame.py --device cuda
    python benchmark_single_frame.py --runs 20
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


def parse_args():
    p = argparse.ArgumentParser(description="Single-frame speed benchmark")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--runs",   type=int, default=10,
                   help="Number of timed forward passes  (default: 10)")
    return p.parse_args()


def find_video(root: Path) -> Path:
    for ext in ("*.mp4", "*.avi", "*.mov"):
        hits = sorted((root / "samples").glob(ext))
        if hits:
            return hits[0]
    raise FileNotFoundError("No video file found in samples/")


def read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def time_runs(fn, n: int) -> list:
    """Run fn() n times and return wall-clock times in milliseconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def report(label: str, times: list) -> None:
    arr = np.array(times)
    print(f"\n  {label}")
    print(f"    runs    : {len(arr)}")
    print(f"    median  : {np.median(arr):7.2f} ms")
    print(f"    mean    : {np.mean(arr):7.2f} ms")
    print(f"    min     : {np.min(arr):7.2f} ms")
    print(f"    max     : {np.max(arr):7.2f} ms")


def main():
    args = parse_args()
    root   = Path(__file__).parent
    device = args.device
    runs   = args.runs

    # ── Find video and read first frame ───────────────────────────────────────
    video_path = find_video(root)
    frame_rgb  = read_first_frame(video_path)
    h, w       = frame_rgb.shape[:2]

    print("=" * 54)
    print("  SINGLE-FRAME SPEED BENCHMARK")
    print("=" * 54)
    print(f"  Video  : {video_path.name}  ({w}×{h})")
    print(f"  Device : {device}")
    print(f"  Runs   : {runs} timed forward passes per stage")

    # ── Stage 1: ResNet50 preprocessing ───────────────────────────────────────
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\n  Loading ResNet50 ...", end=" ", flush=True)
    t0 = time.perf_counter()
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet = nn.Sequential(*list(resnet.children())[:-1])   # strip classifier
    resnet.eval().to(device)
    print(f"done  ({(time.perf_counter()-t0)*1000:.0f} ms)")

    # Pre-convert frame to tensor once (we benchmark the forward pass, not PIL)
    input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)

    def resnet_forward():
        with torch.no_grad():
            feat = resnet(input_tensor)
            # 2048 → 512 (same reduction used in the pipeline)
            feat = feat.squeeze().cpu().numpy().reshape(4, 512).mean(axis=0)
        return feat

    # Warm-up
    resnet_forward()

    times_resnet = time_runs(resnet_forward, runs)
    report("ResNet50 forward pass (1 frame → 512-dim feature)", times_resnet)

    # ── Stage 2: JointCAM inference ───────────────────────────────────────────
    model_path = None
    for name in ("jointcam_finetuned_v4.pt", "jointcam_model.pt"):
        candidate = root / "models" / name
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        print("\n  [SKIP] JointCAM: no model file found in models/")
        print("         Place jointcam_finetuned_v4.pt (or jointcam_model.pt) there to benchmark.")
    else:
        print(f"\n  Loading JointCAM ({model_path.name}) ...", end=" ", flush=True)
        try:
            from test_emotions import load_model
            t0 = time.perf_counter()
            model = load_model(str(model_path), device)
            print(f"done  ({(time.perf_counter()-t0)*1000:.0f} ms)")

            # Minimal 1-frame input: video (1,1,512), audio (1,1,1024)
            # Audio features are zeros — we are timing model speed, not extraction
            video_t = torch.zeros(1, 1, 512,  device=device)
            audio_t = torch.zeros(1, 1, 1024, device=device)

            def jointcam_forward():
                with torch.no_grad():
                    v, a = model(video_t, audio_t)
                return v.cpu(), a.cpu()

            # Warm-up
            jointcam_forward()

            times_joint = time_runs(jointcam_forward, runs)
            report("JointCAM inference (1-frame sequence → valence + arousal)", times_joint)

            # Combined cost per frame
            combined = [r + j for r, j in zip(times_resnet, times_joint)]
            report("ResNet50 + JointCAM combined (1 frame end-to-end)", combined)

        except ImportError:
            print("\n  [SKIP] JointCAM: could not import test_emotions.py")
            print("         Place test_emotions.py in the project root to benchmark.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 54)
    print("  Interpretation guide")
    print("=" * 54)
    med = np.median(times_resnet)
    fps_equiv = 1000.0 / med if med > 0 else float("inf")
    print(f"  ResNet50 median {med:.1f} ms → ~{fps_equiv:.0f} frames/s throughput")
    print(f"  A 3s window at 15 fps = 45 frames")
    print(f"  Estimated ResNet time for full window: {med * 45:.0f} ms  "
          f"({med * 45 / 1000:.1f} s)")
    print()


if __name__ == "__main__":
    main()
