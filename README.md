# VA Driven Behaviors

A stability-first valence–arousal (VA) behavioral adaptation framework for mobile robots.

The system estimates continuous valence and arousal from audio-visual input, applies
robust temporal analysis, and maps the result to one of five behavioral intents
(`DE_ESCALATE`, `CAUTION`, `CHECK_IN`, `ENGAGE`, `NEUTRAL`). The active integration
target is the **Petoi Bittle X** robot dog, controlled over USB/Bluetooth serial.

Both the offline and online pipelines share the same windowed analysis loop and produce
output in identical formats.

**Detailed references:**
- [readme/offline.md](readme/offline.md) — analyzing pre-recorded video files
- [readme/online.md](readme/online.md) — live webcam/microphone streaming sessions
- [robot/bittle.md](robot/bittle.md) — Bittle X adapter: architecture, commands, tuning, debugging

> **Note for repo maintainers:** This file should live at the repository root so GitHub
> renders it automatically. The supporting docs (`offline.md`, `online.md`) live in
> `readme/`, and the adapter reference lives in `robot/`.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

To use the Bittle X robot, also install pyserial:

```bash
pip install pyserial
```

### 2. Place required files

```
va-driven-behaviors/
├── test_emotions.py              ← copy here manually (JointCAM inference)
├── models/
│   └── jointcam_finetuned_v4.pt  ← copy here manually (or jointcam_model.pt)
└── config/
    └── calibration.json          ← generated or provided
```

### 3. Run offline — pre-recorded video, no robot

```bash
python run_offline.py samples/video67.mp4
```

### 4. Run offline — with Bittle X on hardware

```bash
python run_offline.py samples/video67.mp4 --robot-port COM3
```

### 5. Run offline — with Bittle X in mock mode (prints commands, no hardware needed)

```bash
python run_offline.py samples/video67.mp4 --mock-robot
```

### 6. Run online — live webcam + mic, with Bittle X

```bash
python run_online.py --robot-port COM3
```

### 7. Run on Google Colab (GPU — recommended for final results)

Open `Colab_Diagnostic_Tool.ipynb`, set runtime to GPU, and run all cells.

---

## Unified Pipeline Architecture

Both pipelines share the same core analysis loop. The only differences are the
input source and the threading model:

| | Offline | Online |
|---|---|---|
| Input | Pre-recorded video file | Live webcam + microphone ring buffer |
| Windowing | Sequential segment extraction | Pipelined snapshot + worker thread |
| Stability gating | `ReactionHistory` (confidence + cooldown) | Same |
| Robot integration | Direct `robot.apply_reaction()` call | `on_behavior_update` callback |
| Output format | BEHAVIOR DISPATCH + summary table | Same |
| Entry point | `run_offline.py` | `run_online.py` |

### The 3-Second Responsive Window

The system is locked to **3-second analysis windows** by default. Previous long-window
approaches buried emotional micro-behaviors in statistical noise. At 3 seconds the
robot reacts to changes in affect that are visible to a human observer in real time.

### What Each Window Computes

Every window goes through nine stages:

1. **Feature extraction** — JointCAM encodes video frames and audio into latent features.
2. **Alignment** — video and audio streams are resampled to the same time axis.
3. **Inference** — the model outputs `valence_series[T]` and `arousal_series[T]`.
4. **Robust preprocessing** — winsorize → Hampel outlier filter → rolling-median smoothing.
5. **Session baseline** — trimmed-mean estimator produces stable scalar V and A values.
6. **Trend analysis** — Theil-Sen robust slope detects rising/falling/stable affect.
7. **Volatility** — Median Absolute Deviation (MAD) quantifies signal instability.
8. **State classification** — percentile-based thresholds assign one of 8 VA state labels.
9. **Reaction mapping** — a volatility-gated 5-intent policy produces a `ReactionAction`.

### Volatility-Gated 5-Intent Policy

Volatility is checked first and takes priority over the VA state label:

| Condition | Intent | Robot behavior (Bittle X) |
|---|---|---|
| volatility ≥ 0.25 | **DE_ESCALATE** | Backs up, then sits |
| 0.15 ≤ volatility < 0.25 | **CAUTION** | Rigid balance stance |
| negative-high-arousal | **DE_ESCALATE** | Backs up, then sits |
| negative-low-arousal | **CHECK_IN** | Nods |
| positive-high-arousal | **ENGAGE** | Waves |
| neutral | **NEUTRAL** | Standing balance |

**Pose mode** is selected by confidence:
- `state_confidence ≥ 0.6` and low volatility → **VA_POSE** (expressive, model-driven)
- Otherwise → **REACTION_POSE** (conservative, rule-based)

### Stability Gating

Before any intent change is committed, two sequential gates prevent jitter:

- **Confidence gate** — new intent rejected if `state_confidence < 0.55` (default).
  Falls back to `NEUTRAL`.
- **Cooldown gate** — new intent rejected if fewer than `0.5 s` have elapsed since
  the last accepted change (default). Previous intent is maintained.

Both thresholds are adjustable via `--min-confidence` and `--cooldown`.

### Robot Integration — Bittle X Queue

The adapter adds a **single-slot queue** between the pipeline and the robot. If the
robot is busy executing a motion, the incoming intent is held in the pending slot.
If another intent arrives before the robot finishes, the newer intent replaces the
stale one — the robot always picks up the most recent emotional state rather than
replaying an outdated backlog.

See [robot/bittle.md](robot/bittle.md) for the full architecture, tuning guide, and
debugging checklist.

---

## Flag Reference

### Shared flags (both `run_offline.py` and `run_online.py`)

| Flag | Default | Description |
|---|---|---|
| `--window S` | `3.0` | Analysis window in seconds |
| `--sparse` | off | 2 frames per window — fast CPU testing |
| `--device cpu\|cuda` | `cpu` | Torch inference device |
| `--model PATH` | auto | Path to `.pt` model file |
| `--min-confidence C` | `0.55` | Confidence gate threshold |
| `--cooldown S` | `0.5` | Cooldown gate in seconds |
| `--debug` | off | Verbose per-stage output |
| `--robot-port PORT` | off | Serial port for Bittle X (e.g. `COM3`, `/dev/ttyUSB0`) |
| `--mock-robot` | off | Print robot commands to terminal without hardware |

### Offline-only flags

| Flag | Default | Description |
|---|---|---|
| `video_path` | — | Path to input video file (required) |
| `--calibration PATH` | `config/calibration.json` | Calibration file path |
| `--audio PATH` | auto | Separate audio file |

### Online-only flags

| Flag | Default | Description |
|---|---|---|
| `--duration S` | `30.0` | Total session duration in seconds |
| `--camera N` | `0` | OpenCV camera index |
| `--fps F` | `15.0` | Webcam capture frame rate |
| `--no-audio` | off | Disable microphone capture |
| `--no-cleanup` | off | Keep temp snapshot files after session |

See [readme/offline.md](readme/offline.md) and [readme/online.md](readme/online.md) for
complete per-pipeline flag tables and example invocations.

---

## Performance Tiers

| Tier | Command | Frames/window | Time/window |
|---|---|---|---|
| CPU sparse (rapid verify) | `--sparse` | 2 | ~5 s |
| CPU full | *(default on CPU)* | 90 | ~40–50 s |
| GPU full (Colab T4) | `--device cuda` | 90 | ~2–3 s |

GPU processes **45× more visual data** than CPU sparse mode while running **2× faster**.
For final results and Bittle integration testing, use GPU.

---

## Running Tests

```bash
python tests/test_robust_pipeline.py
```

Expected: 8 tests, all PASS. No model or video files required.

---

## Regenerating Calibration

```bash
python pipeline/generate_calibration.py samples/ \
    --model_path models/jointcam_finetuned_v4.pt \
    --output config/calibration.json \
    --device cpu
```

---

## Folder Structure

```
va-driven-behaviors/
│
├── README.md                      ← This file — move to repo root for GitHub rendering
│
├── readme/                        ← Detailed pipeline references
│   ├── README.md                  ← (same file — lives here currently)
│   ├── offline.md                 ← Offline pipeline detailed reference
│   └── online.md                  ← Online pipeline detailed reference
│
├── robot/                         ← Hardware adapter
│   ├── bittle_adapter.py          ← Bittle X serial adapter (queue, guards, serial I/O)
│   └── bittle.md                  ← Adapter architecture, commands, tuning, debugging
│
├── requirements.txt
├── setup_paths.py
├── test_emotions.py               ← PLACE HERE MANUALLY (JointCAM inference)
│
├── run_offline.py                 ← Analyze a pre-recorded video (windowed)
├── run_online.py                  ← Live webcam/mic streaming session
│
├── pipeline/                      ← Core VA analysis (no hardware dependencies)
│   ├── __init__.py
│   ├── emotion_analyzer.py        ← Preprocessing, baseline, trends, classification
│   ├── robust_stats.py            ← MAD, IQR, Hampel filter, trimmed mean, winsorize
│   ├── reaction_action.py         ← ReactionAction dataclass (structured output schema)
│   ├── spot_reaction_mapper.py    ← Maps VA state → ReactionAction
│   ├── intent_selector.py         ← 5-intent volatility-gated policy
│   ├── affect_filter.py           ← EMA smoothing + rate limiting + hysteresis
│   └── generate_calibration.py   ← Tool: generate calibration.json from a video set
│
├── online/                        ← Live capture and session orchestration
│   ├── __init__.py
│   ├── live_capture.py            ← Records webcam + microphone to temp files
│   ├── window_analyzer.py         ← Runs the full 9-stage pipeline on one window
│   ├── online_session.py          ← ReactionHistory (stability gating)
│   └── streaming_session.py       ← Ring buffer + pipelined StreamingSession
│
├── models/                        ← PLACE MODEL .pt FILES HERE (see models/README.md)
├── config/
│   └── calibration.json
├── samples/                       ← PLACE TEST VIDEOS HERE (see samples/README.md)
├── tests/
│   └── test_robust_pipeline.py
└── Colab_Diagnostic_Tool.ipynb
```

---

## Dependencies

| Package | Used for |
|---|---|
| `numpy` | Array operations throughout pipeline |
| `scipy` | Theil-Sen slope estimator in trend analysis |
| `torch` | JointCAM model inference |
| `opencv-python` (`cv2`) | Video reading (offline) + webcam capture (online) |
| `sounddevice` | Microphone capture for online mode (optional) |
| `pyserial` | Bittle X serial communication (optional — mock mode if absent) |
| `test_emotions` | JointCAM feature extraction + inference (place in project root) |

---

## Known Issues

**Audio extraction warnings** — "Could not extract audio" appears on some Windows/Linux
environments due to `librosa`/`ffmpeg` path issues. The pipeline uses zero audio
features as a fallback and continues with visual-only analysis. Ensure `ffmpeg` is on
your system PATH to resolve it.

**No `[BITTLE]` output** — if the robot is connected but no commands appear, the most
common cause is a missing `--robot-port` or `--mock-robot` flag. The adapter is only
initialised when one of those flags is present. See the debugging checklist in
[robot/bittle.md](robot/bittle.md).

---

## Research Context

> **Valence-Arousal-Driven Behavior Adaptation for Mobile Robots**
> Vanohra Gaspard, Ikechukwu Anyanwu, Vignesh A. M. Raja, Anu G. Bourgeois, Ashwin Ashok
> Georgia State University

Supported by U.S. Army Research Laboratory (ARL W911NF-23-2-0224).

The core design philosophy: **respond to persistent affective patterns, not transient
fluctuations.** Volatility gating, confidence thresholds, temporal trend analysis, and
momentum tracking are the mechanisms that enforce this stability-first approach.
