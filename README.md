# VA Driven Behaviors

A stability-first valence–arousal (VA) behavioral adaptation framework for mobile robots.

The system estimates continuous valence and arousal from audio-visual input, applies
robust temporal analysis, and maps the result to one of five behavioral intents
(DE_ESCALATE, CAUTION, CHECK_IN, ENGAGE, NEUTRAL).  All outputs are verified through
the terminal.  No simulation software is required.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place required files in the project root

Copy the following files directly into the `VA driven behaviors/` folder (same level as `run_offline.py`):

```
VA driven behaviors/
├── test_emotions.py              ← copy here manually
├── models/
│   └── jointcam_finetuned_v4.pt  ← copy here manually (or jointcam_model.pt)
└── ...
```

> **test_emotions.py** contains the JointCAM feature extraction and inference functions.  
> **The model `.pt` file** is the JointCAM checkpoint.

### 3. Run offline (pre-recorded video)

```bash
python run_offline.py path/to/video.mp4 --windowed --sparse
```

### 4. Run online (live webcam + mic, streaming mode)

```bash
python run_online.py
```

### 5. Running on Google Colab (GPU)

For the fastest and most accurate results (analysis in < 5s), use the included:
**`Colab_Diagnostic_Tool.ipynb`**

1. Upload the project zip and model to Colab.
2. Open the notebook and set Runtime to **GPU**.
3. Run the setup and analysis cells.

---

### 6. Technical Flag Dictionary (Concise)

- `--windowed`: Enables the **3.0s behavior loop** (Professor's recommendation).
- `--sparse`: **CPU Optimization**. Analyzes only 2 frames per window. Use for rapid local testing.
- `--device cuda`: **GPU Acceleration**. Use on Colab to analyze all 90 frames for maximum accuracy.

---

## 🚀 Performance Tiers (CPU vs. GPU)

The pipeline is designed to be hardware-flexible:
1.  **CPU Tier ("Rapid Verify")**: Uses `--sparse` to provide instant feedback. The CPU is **NOT** a bottleneck; it is a high-speed diagnostic tool for logic verification.
2.  **GPU Tier ("High Fidelity")**: Analyzes 100% of the video data. Use this for the final SIGDIAL data generation and momentum validation.

---

## ⚠️ Known Issue: Audio Extraction

We are currently experiencing consistent "Warning: Could not extract audio" errors in both Windows and Linux (Colab) environments. 
- **Current Status**: The system uses a "Modality-Robust" fallback (zeros) and continues the analysis using visual data.
- **Note for Teammate**: Please investigate the `librosa` / `ffmpeg` pathing or consider a `moviepy` wrapper if high-fidelity audio features are required for the final submission.

---

## How to Run — Offline Mode

Analyze a pre-recorded video file:

```bash
# Basic
python run_offline.py samples/video67.mp4

# With debug output (prints per-stage detail)
python run_offline.py samples/video67.mp4 --debug

# Specify model file explicitly
python run_offline.py samples/video67.mp4 --model models/jointcam_finetuned_v4.pt

# Use GPU
python run_offline.py samples/video67.mp4 --device cuda

# Separate audio file
python run_offline.py samples/video67.mp4 --audio path/to/audio.wav
```

### Expected offline output

```
Calibration loaded: calibration.json
Loading model: jointcam_finetuned_v4.pt ... done.
Running VA analysis...

══════════════════════════════════════════════════════════════
  VA ANALYSIS RESULTS
══════════════════════════════════════════════════════════════

  VA Baseline:
    Valence:  +0.3214  | Trend: rising       | Volatility: 0.0821
    Arousal:  +0.5634  | Trend: stable       | Volatility: 0.0644

  State Classification:
    VA Label:    positive-high-arousal
    Confidence:  0.7823
    Volatility:  0.0821

  Reaction Mapping:
    Intent:      ENGAGE
    Pose mode:   VA_POSE
    Speed mult:  1.10x
    Dist. mult:  0.80x
    Duration:    1.5s
    Explanation: Positive affect with high arousal; approach and engage.
```

---

## How to Run — Online Mode

Run a live 30-second streaming session from webcam + microphone:

```bash
# Default: 30s session, 3 × 10s windows
python run_online.py

# With debug output
python run_online.py --debug

# Video-only (no microphone)
python run_online.py --no-audio

# Different camera index
python run_online.py --camera 1

# Custom duration / window size (e.g., 60s session with 15s windows = 4 windows)
python run_online.py --duration 60 --window 15

# Adjust stability gates
python run_online.py --min-confidence 0.6 --cooldown 6

# Keep temp snapshot files after session (for debugging)
python run_online.py --no-cleanup
```

### How the streaming session works

The online pipeline uses a **ring buffer + pipelined architecture**.  The camera
never stops — it runs as a dedicated thread for the full session duration, writing
every frame into a fixed-size ring buffer that always holds the last `window_dur`
seconds of footage and audio.

```
t=0s   camera starts, ring buffer fills continuously ──────────────────────►
t=10s  snapshot 1 taken ──► worker 1 analyzes (camera keeps rolling)
t=20s  snapshot 2 taken ──► worker 2 analyzes (camera keeps rolling)
t=30s  snapshot 3 taken ──► worker 3 analyzes; session timer ends
                             executor drains remaining workers
                             session summary printed
```

Each snapshot is a frozen copy of the ring buffer at that moment — the worker
thread never touches the live buffer.  Analysis therefore runs in parallel with
the next capture window rather than blocking it.

**When each worker finishes, behavior is dispatched immediately:**

```
Worker finishes
  → ┌─ BEHAVIOR DISPATCH ─────────────────────────────┐
    │  Intent :  ENGAGE        (CHANGED)               │
    │  Pose   :  VA_POSE       speed=1.10×  dist=0.80× │
    │  State  :  positive-high-arousal  (conf: 0.600)  │
    │  VA     :  valence=+0.386   arousal=+0.162        │
    └──────────────────────────────────────────────────┘
  → detailed VA analysis printed
  → result logged to session_results
```

The dispatch block is always the **first output** from a finished worker.
For robot hardware integration, pass `on_behavior_update=my_callback` to
`StreamingSession` — it is called with `(intent, reaction_action, analysis)`
at the same instant the dispatch block prints.

---

### Analysis Latency vs Window Duration

> **This is the most important operational constraint for online deployment.**

The pipeline must analyze video + audio using a deep model (JointCAM) on each
snapshot.  On CPU, this typically takes **40–50 seconds** for a 10-second window.
That gap — between when a snapshot is taken and when the behavior decision is
emitted — is the **behavioral latency** of the system.

**What happens in your session:**

| Event | Wall clock | Behavioral latency |
|---|---|---|
| Snapshot 1 taken at t=10s | t=10s | — |
| Decision from snapshot 1 emitted | t≈61s | **~51 seconds late** |
| Snapshot 2 taken at t=20s | t=20s | — |
| Decision from snapshot 2 emitted | t≈61s | **~41 seconds late** |

At 41–51 seconds of latency, the system is reacting to how the person felt
**almost a minute ago**.  In a live interaction, that renders the behavior
output functionally stale.

**Why does overflow happen?**

The ring buffer captures `window_dur × 1.5` seconds of data (the 1.5× factor
absorbs timer jitter).  A 10-second window therefore produces **15 seconds of
video** (225 frames at 15 fps) per snapshot — 50% more data than the window
name suggests.  Combined with CPU inference, analysis time far exceeds the
window period.

**Overflow cascade:**

```
t=10s  Worker 1 starts  (will take ~51s to finish)
t=20s  Worker 2 starts  ← WARNING: Worker 1 still running
t=30s  Worker 3 starts  ← both previous workers still running
         ↑ executor has max_workers=2, so Worker 3 queues behind Worker 2
         session timer ends; executor.shutdown(wait=True) blocks here
t≈61s  Workers 1 + 2 finish roughly simultaneously
t≈71s  Worker 3 finishes
         session summary finally prints
```

**How to fix it — by hardware:**

| Hardware | Recommended `--window` | Expected analysis time |
|---|---|---|
| CPU only (current) | `--window 60` | ~45–55s |
| GPU (CUDA) | `--window 10` | ~3–8s |
| GPU + `--no-audio` | `--window 10` | ~2–5s |

**How to fix it — by arguments (CPU):**

```bash
# Widen the window so analysis fits comfortably
python run_online.py --duration 180 --window 60   # 3 × 60s windows, ~45s analysis

# Disable audio to halve the pipeline work
python run_online.py --window 60 --no-audio

# Use GPU if available
python run_online.py --device cuda
```

**Analysis timing** is printed for every window:

```
  Analysis time: 7.43s          ← healthy: fits within 10s window
  Analysis time: 41.03s  ***    ← overflow: 4× longer than window
```

If analysis takes longer than `window_dur`, a warning is shown and an `!` flag
appears in the session summary table.

Between-window behavior changes are gated by a **confidence threshold**
(default 0.55) and a **cooldown** (default 8s) to prevent jittery switching.

---

#### Combined Architecture: Ring Buffer + Pipelined Analysis

### The Core Idea
Think of it as two completely independent loops running simultaneously:

1. Loop A (Capture): Never stops. Continuously writes incoming frames/audio into a ring buffer.
2. Loop B (Analysis): Wakes up on a timer, snapshots the buffer, and runs the pipeline — while Loop A keeps going uninterrupted.

Neither loop waits for the other.

### How the Ring Buffer Works
A ring buffer is a fixed-size structure (e.g., "last 10 seconds of frames"). As new frames come in, old ones fall off the back. Think of it like a conveyor belt — it's always moving, always full, always current.

time →
[f1][f2][f3][f4][f5][f6][f7][f8][f9][f10]  ← always the last 10s
                                  ↑ new frame pushes f1 off


The buffer is shared memory between Loop A and Loop B. Loop A writes to it; Loop B reads from it.

### How They Work Together, Step by Step
t=0: Loop A starts. Capture begins. Frames flow into the ring buffer continuously.

t=0 to t=10: Buffer fills up. Loop B hasn't fired yet.

t=10: A timer fires. Loop B wakes up, takes a snapshot (a frozen copy) of the current buffer contents, and hands it off to a worker thread for analysis. Loop A does not pause — it keeps writing new frames into the live buffer.

t=10 to t=20: Loop A continues filling the live buffer with new frames. Loop B's worker thread is analyzing the t=0–10 snapshot in parallel.


PS [PROJECT_ROOT]> python run_online.py
Warning: Missing 'std' for 'valence' in calibration file. Using fallback thresholds.
Warning: Using fallback calibration thresholds.
Loading model: jointcam_finetuned_v4.pt ... Loading model from: [PROJECT_ROOT]\models\jointcam_finetuned_v4.pt
  Loaded 77/85 parameters
done.

==============================================================
  VA ONLINE PIPELINE — STREAMING SESSION
  (Ring Buffer + Pipelined Architecture)
==============================================================
  Duration:  30s  (3 windows × 10s)
  Camera:    index 0  |  FPS: 15.0
  Audio:     enabled
  Device:    cpu
  Model:     jointcam_finetuned_v4.pt


╔════════════════════════════════════════════════════════════════╗
║                      VA STREAMING SESSION                      ║
║                   (Ring Buffer + Pipelined)                    ║
╚════════════════════════════════════════════════════════════════╝
  Started:      2026-04-01 18:22:01
  Duration:     30s  |  3 windows × 10s each
  Mode:         Camera never pauses — analysis runs in parallel
  Overflow:     warned if analysis takes longer than 10s
  Confidence:   0.55  |  Cooldown: 8s

Camera opened: 640×360 @ 30.0 fps  (requested 320×240 @ 15.0)
  Camera running.  First analysis snapshot in 10s.

──────────────────────────────────────────────────────────────────
  WINDOW 1/3  |  t≈10s  |  18:22:11  |  225 frames in buffer
──────────────────────────────────────────────────────────────────
  [Window 1] Running VA analysis pipeline...
  Extracting video features from: [TEMP_DIR]\va_stream_0.avi
    Video: 225 frames, 15.0 FPS, 15.00s duration

──────────────────────────────────────────────────────────────────
  WINDOW 2/3  |  t≈20s  |  18:22:21  |  225 frames in buffer
  [WARNING] Worker overflow: previous analysis still running when this snapshot fired.  Consider increasing --window.
──────────────────────────────────────────────────────────────────
  [Window 2] Running VA analysis pipeline...
  Extracting video features from: [TEMP_DIR]\va_stream_1.avi
    Video: 225 frames, 15.0 FPS, 15.00s duration

──────────────────────────────────────────────────────────────────
  WINDOW 3/3  |  t≈30s  |  18:22:31  |  225 frames in buffer
  [WARNING] Worker overflow: previous analysis still running when this snapshot fired.  Consider increasing --window.
──────────────────────────────────────────────────────────────────
    Extracted 225 frame features, shape: (225, 512)
  Extracting audio features from: [TEMP_DIR]\va_stream_0.wav
    Audio: 9.44s duration, 16000 Hz sample rate
    Extracted 300 audio frames, shape: (300, 128)

  ┌─ BEHAVIOR DISPATCH  [window 1  t≈10s] ────────────────────────────────────────────────────┐
  │  Intent :  ENGAGE        (CHANGED)
  │  Pose   :  REACTION_POSE   speed=0.70×  dist=1.00×  dur=1.5s
  │  State  :  positive-high-arousal  (conf: 0.550)
  │  VA     :  valence=+0.302   arousal=+0.258
  └──────────────────────────────────────────────────────────────────────────┘

  Snapshot: 225 frames  |  Audio: mic
  Analysis time: 39.60s  *** exceeds window (10s) — worker overflow likely next!

  VA Baseline:
    Valence:  +0.302  | Trend: mixed        | Volatility: 0.007
    Arousal:  +0.258  | Trend: rising       | Volatility: 0.007

  State:
    Label:      positive-high-arousal
    Confidence: 0.550   Volatility: 0.007
    Pose mode:  REACTION_POSE

  ★  BEHAVIOR UPDATE:  —  →  ENGAGE  (first window)
  [Window 3] Running VA analysis pipeline...
  Extracting video features from: [TEMP_DIR]\va_stream_2.avi
    Video: 225 frames, 15.0 FPS, 15.00s duration
    Extracted 225 frame features, shape: (225, 512)
  Extracting audio features from: [TEMP_DIR]\va_stream_1.wav
    Audio: 14.98s duration, 16000 Hz sample rate
    Extracted 300 audio frames, shape: (300, 128)

  ┌─ BEHAVIOR DISPATCH  [window 2  t≈20s] ────────────────────────────────────────────────────┐
  │  Intent :  NEUTRAL       (maintained)
  │  Pose   :  REACTION_POSE   speed=0.70×  dist=1.00×  dur=1.5s
  │  State  :  positive-high-arousal  (conf: 0.500)
  │  VA     :  valence=+0.293   arousal=+0.267
  └──────────────────────────────────────────────────────────────────────────┘

  Snapshot: 225 frames  |  Audio: mic
  Analysis time: 45.43s  *** exceeds window (10s) — worker overflow likely next!

  VA Baseline:
    Valence:  +0.293  | Trend: mixed        | Volatility: 0.008
    Arousal:  +0.267  | Trend: mixed        | Volatility: 0.003

  State:
    Label:      positive-high-arousal
    Confidence: 0.500   Volatility: 0.008
    Pose mode:  REACTION_POSE

  ·  Maintaining:  NEUTRAL  (confidence < 0.55)
    Extracted 225 frame features, shape: (225, 512)
  Extracting audio features from: C:\Users\vanoh\AppData\Local\Temp\va_stream_2.wav
    Audio: 14.98s duration, 16000 Hz sample rate
    Extracted 300 audio frames, shape: (300, 128)

  ┌─ BEHAVIOR DISPATCH  [window 3  t≈30s] ────────────────────────────────────────────────────┐
  │  Intent :  NEUTRAL       (maintained)
  │  Pose   :  REACTION_POSE   speed=0.70×  dist=1.00×  dur=1.5s
  │  State  :  positive-high-arousal  (conf: 0.500)
  │  VA     :  valence=+0.315   arousal=+0.256
  └──────────────────────────────────────────────────────────────────────────┘

  Snapshot: 225 frames  |  Audio: mic
  Analysis time: 35.10s  *** exceeds window (10s) — worker overflow likely next!

  VA Baseline:
    Valence:  +0.315  | Trend: mixed        | Volatility: 0.007
    Arousal:  +0.256  | Trend: mixed        | Volatility: 0.003

  State:
    Label:      positive-high-arousal
    Confidence: 0.500   Volatility: 0.007
    Pose mode:  REACTION_POSE

  ·  Maintaining:  NEUTRAL  (confidence < 0.55)

╔════════════════════════════════════════════════════════════════╗
║                        SESSION COMPLETE                        ║
╚════════════════════════════════════════════════════════════════╝
  Total time: 85.1s

  Behavior timeline:
  Win       t  VA Label                    Conf   Analysis  Intent           Note
  ────────────────────────────────────────────────────────────────────────────────
  1        10s  positive-high-arousal       0.55    39.60s !  ENGAGE           ← changed
  2        20s  positive-high-arousal       0.50    45.43s !  NEUTRAL
  3        30s  positive-high-arousal       0.50    35.10s !  NEUTRAL

  Note: '!' in the Analysis column means that window's analysis
        exceeded the 10s window duration (worker overflow risk).

  Done.

## Regenerating Calibration

If you update the model or run on a new dataset, regenerate calibration:

```bash
python pipeline/generate_calibration.py samples/ \
    --model_path models/jointcam_finetuned_v4.pt \
    --output config/calibration.json \
    --device cpu
```

This processes all videos in `samples/` and writes updated percentile statistics to
`config/calibration.json`. The pipeline uses these to set scale-aware thresholds for
trend detection, volatility gating, and state classification.

---

## Running Tests

Tests do **not** require a model or any video files — they only test the analysis math:

```bash
python tests/test_robust_pipeline.py
```

Expected output: 8 tests, all PASS.

---

## Folder Structure

```
VA driven behaviors/
│
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── setup_paths.py             ← No-op stub (no configuration needed)
│
├── test_emotions.py           ← PLACE HERE MANUALLY (JointCAM inference)
│
├── run_offline.py             ← Analyze a pre-recorded video file
├── run_online.py              ← Live 30-second webcam/mic session
│
├── pipeline/                  ← Core VA analysis (no simulation dependencies)
│   ├── __init__.py
│   ├── emotion_analyzer.py    ← Preprocessing, baseline, trends, classification
│   ├── robust_stats.py        ← MAD, IQR, Hampel filter, trimmed mean, winsorize
│   ├── reaction_action.py     ← ReactionAction dataclass (structured output schema)
│   ├── spot_reaction_mapper.py ← Maps VA state → ReactionAction
│   ├── intent_selector.py     ← 5-intent volatility-gated policy
│   ├── affect_filter.py       ← EMA smoothing + rate limiting + hysteresis
│   └── generate_calibration.py ← Tool: generate calibration.json from a video set
│
├── online/                    ← Live capture and session orchestration
│   ├── __init__.py
│   ├── live_capture.py        ← Records webcam + microphone to temp files
│   ├── window_analyzer.py     ← Runs the full pipeline on one video window
│   └── online_session.py      ← 30-second session loop with hysteresis
│
├── models/                    ← PLACE MODEL .pt FILES HERE
│   └── README.md              ← Instructions for which files go here
│
├── config/
│   └── calibration.json       ← VA scale calibration statistics
│
├── samples/                   ← Optional: test videos for offline mode
│   └── README.md
│
├── tests/
│   ├── __init__.py
│   └── test_robust_pipeline.py ← Unit tests (no model needed)
│
└── legacy_check/              ← Simulation-dependent code kept for reference only
    ├── README.md
    ├── simulation/
    └── tools/
```

---

## What the Pipeline Does

Rather than reacting to every frame, the system:

1. **Estimates VA** — JointCAM produces per-frame valence and arousal values.
2. **Preprocesses robustly** — winsorization, Hampel outlier filtering, and EMA smoothing remove noise.
3. **Computes a session baseline** — trimmed-mean estimator produces a stable "emotional center" for the clip.
4. **Analyzes trends** — Theil-Sen slope + start/end median delta detect whether affect is rising, falling, stable, mixed, or uncertain.
5. **Estimates volatility** — Median Absolute Deviation (MAD) quantifies signal instability.
6. **Classifies VA state** — percentile-based thresholds assign one of 8 state labels (e.g., `positive-high-arousal`, `neutral`).
7. **Maps to intent** — a volatility-gated 5-intent policy selects the robot's behavioral intent and pose mode.

The robot responds to **sustained patterns**, not instantaneous noise.

---

## Offline vs. Online Modes

| | Offline | Online |
|---|---|---|
| Input | Pre-recorded video file | Live webcam + microphone |
| Duration | Full video (any length) | 30-second session (configurable) |
| Analysis frequency | Once (whole video) | Every 10 seconds (3 windows) |
| Entry point | `run_offline.py` | `run_online.py` |
| Stability mechanism | Robust preprocessing | + confidence gate + cooldown |

---

## Dependencies

| Package | Used for |
|---|---|
| `numpy` | Array operations throughout pipeline |
| `scipy` | Theil-Sen slope estimator in trend analysis |
| `torch` | JointCAM model inference |
| `opencv-python` (`cv2`) | Video reading (offline) + webcam capture (online) |
| `sounddevice` | Microphone capture for online mode (optional) |
| `test_emotions` | JointCAM feature extraction + inference (place in project root) |

---

## Research Context

This project is part of the paper:

> **Valence-Arousal-Driven Behavior Adaptation for Mobile Robots**  
> Vanohra Gaspard, Ikechukwu Anyanwu, Vignesh A. M. Raja, Anu G. Bourgeois, Ashwin Ashok  
> Georgia State University

Supported by U.S. Army Research Laboratory (ARL W911NF-23-2-0224).

The core design philosophy: **respond to persistent affective patterns, not
transient fluctuations.**  Volatility gating, confidence thresholds, and temporal
trend analysis are the mechanisms that enforce this stability-first approach.
