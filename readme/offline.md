# Offline Pipeline — Detailed Reference

`run_offline.py` analyzes a pre-recorded video file using the same windowed loop,
stability gating, and output format as the online pipeline. Every `--window` seconds
of video is treated as one analysis window and processed identically to a live snapshot.

---

## How It Works

### Windowed Analysis Loop

The video is sliced into non-overlapping segments of `--window` seconds (default 3 s).
Each segment is written to a temp file, run through the full VA pipeline, and then
deleted. Windows are processed sequentially; momentum (last valence/arousal) is
chained from one window to the next.

```
video.mp4  (e.g. 30s)
  │
  ├── window 1  [0s – 3s]   → extract → analyze → gate → dispatch → robot
  ├── window 2  [3s – 6s]   → extract → analyze → gate → dispatch → robot
  ├── window 3  [6s – 9s]   → extract → analyze → gate → dispatch → robot
  ├── ...
  └── window N  [(N-1)×3s – 30s]
```

Final segments shorter than 1 second are skipped.

### Per-Window Pipeline (9 stages)

1. **Feature extraction** — `test_emotions.extract_video_features()` samples frames
   from the segment and passes them through JointCAM's visual encoder.
   `extract_audio_features()` computes mel-spectrogram features from the audio track.

2. **Alignment** — `align_features()` resamples whichever modality has fewer time
   steps to match the other, producing synchronized `[T, D_v]` and `[T, D_a]` tensors.

3. **Inference** — `predict_emotions()` runs the aligned features through JointCAM and
   returns `valence_series[T]` and `arousal_series[T]` in approximately [−1, +1].

4. **Robust preprocessing** — applied independently to each dimension:
   - **Winsorize** — clips the bottom/top 1 % of values to suppress extremes.
   - **Hampel filter** — replaces points more than 3 MADs from the local rolling
     median (window = 7) with that median, handling sudden spikes.
   - **Rolling-median smoothing** — window = 5 frames to reduce residual noise.

5. **Session baseline** — trimmed-mean estimator (trim 10 % each tail) of the
   preprocessed series. This is the reported scalar `valence` / `arousal`.

6. **Trend analysis** — Theil-Sen robust slope over the series. Reports
   `valence_direction` (`rising` / `falling` / `stable` / `mixed`) and
   `valence_delta` (start-to-end median change).

7. **Volatility** — Median Absolute Deviation (MAD) of the preprocessed series.
   `volatility = max(valence_volatility, arousal_volatility)`.

8. **State classification** — calibration-percentile thresholds (p30 / p70 for each
   dimension) assign one of 8 labels:
   `positive-high-arousal`, `positive-low-arousal`,
   `negative-high-arousal`, `negative-low-arousal`,
   `neutral-high-arousal`, `neutral-low-arousal`,
   `neutral-valence-neutral-arousal`, `neutral`.
   A `state_confidence` score accompanies every label.

9. **Reaction mapping** — `SpotReactionMapper.map_to_action()` converts the state
   label, trends, volatility, and confidence into a `ReactionAction`:

   | Condition | Intent | Pose mode |
   |---|---|---|
   | volatility ≥ 0.25 | DE_ESCALATE | REACTION_POSE |
   | 0.15 ≤ volatility < 0.25 | CAUTION | REACTION_POSE |
   | negative-high-arousal | DE_ESCALATE | VA_POSE if conf ≥ 0.6 |
   | negative-low-arousal | CHECK_IN | VA_POSE if conf ≥ 0.6 |
   | positive-high-arousal | ENGAGE | VA_POSE if conf ≥ 0.6 |
   | neutral | NEUTRAL | VA_POSE if conf ≥ 0.6 |

   **Momentum**: if valence or arousal changed by more than 15 % relative to the
   previous window, assertive adjustments are applied to the ENGAGE parameters
   (speed +0.1, duration +0.3 s).

### Stability Gating (`ReactionHistory`)

Before a proposed intent is accepted, two gates are applied — identical to the online
pipeline:

- **Confidence gate** — reject if `state_confidence < --min-confidence` (default 0.55).
  Falls back to `NEUTRAL`.
- **Cooldown gate** — reject if fewer than `--cooldown` seconds have elapsed since the
  last accepted intent change (default 0.5 s). Maintains the previous intent.

The **effective intent** (the stability-gated output) is what gets dispatched to the
robot. The raw model intent (`ra.intent`) is overridden with `effective` before
`apply_reaction()` is called, so the robot always acts on the filtered result.

---

## Running the Offline Pipeline

### Without robot (terminal output only)

```bash
python run_offline.py samples/video67.mp4
```

### With Bittle X — hardware on serial port

```bash
python run_offline.py samples/video67.mp4 --robot-port COM3
```

Replace `COM3` with your actual port. On Linux/macOS use `/dev/ttyUSB0` or
`/dev/cu.usbserial-*`. To find the port: open Device Manager (Windows) or run
`ls /dev/tty*` (Linux/macOS) before and after plugging in the robot.

### With Bittle X — mock mode (no hardware required)

```bash
python run_offline.py samples/video67.mp4 --mock-robot
```

Prints all serial commands to the terminal with `[BITTLE-MOCK]` prefix.
Useful for testing the full pipeline integration without the physical robot.

### Common invocations

```bash
# Standard run — 3s windows, CPU, no robot
python run_offline.py samples/video67.mp4

# With robot on COM3, standard settings
python run_offline.py samples/video67.mp4 --robot-port COM3

# Mock robot — verify intent dispatch without hardware
python run_offline.py samples/video67.mp4 --mock-robot

# GPU, full-depth analysis (recommended for final results)
python run_offline.py samples/video67.mp4 --device cuda --robot-port COM3

# Rapid local verification — 2 frames per window
python run_offline.py samples/video67.mp4 --sparse --mock-robot

# Wider windows (useful for longer videos or CPU-only systems)
python run_offline.py samples/video67.mp4 --window 10 --robot-port COM3

# Stricter stability gates
python run_offline.py samples/video67.mp4 --min-confidence 0.6 --cooldown 2 --robot-port COM3

# Verbose stage-by-stage output
python run_offline.py samples/video67.mp4 --debug --mock-robot
```

---

## Full Flag Reference

| Flag | Default | Description |
|---|---|---|
| `video_path` | — | Path to input video file **(required)** |
| `--window S` | `3.0` | Analysis window size in seconds |
| `--sparse` | off | Process only 2 frames per window — fast CPU testing |
| `--device cpu\|cuda` | `cpu` | Torch inference device |
| `--model PATH` | auto | Path to `.pt` model file |
| `--calibration PATH` | `config/calibration.json` | Path to calibration file |
| `--audio PATH` | auto | Separate audio file (defaults to video audio track) |
| `--min-confidence C` | `0.55` | Minimum confidence to accept an intent change |
| `--cooldown S` | `0.5` | Minimum seconds between behavior changes |
| `--debug` | off | Print per-stage detail (feature shapes, volatility, etc.) |
| `--robot-port PORT` | off | Serial port for Petoi Bittle X (e.g. `COM3`, `/dev/ttyUSB0`) |
| `--mock-robot` | off | Enable mock mode: print commands instead of sending to hardware |

**Notes:**
- `--robot-port` and `--mock-robot` are mutually optional. Using `--mock-robot` without
  `--robot-port` will use `COM7` as the port name in log messages but will never open it.
- If neither flag is passed, `robot = None` and no `[BITTLE]` output will appear.
- `--sparse` and `--debug` can be combined freely.

---

## Expected Output

```
Calibration loaded: calibration.json
Loading model: jointcam_finetuned_v4.pt ... done.

╔════════════════════════════════════════════════════════════════╗
║                      VA OFFLINE PIPELINE                       ║
║           (Windowed — same analysis loop as online)            ║
╚════════════════════════════════════════════════════════════════╝
  Started:     2026-04-15 14:58:15
  Video:       video67.mp4
  Model:       jointcam_finetuned_v4.pt
  Device:      cpu
  Windows:     20 × 3s each
  Confidence:  0.55  |  Cooldown: 0.5s

[BITTLE] Connected on COM3 @ 115200 baud.
[BITTLE] → kbalance
  Robot:       Petoi Bittle X (hardware on COM3)

──────────────────────────────────────────────────────────────────
  WINDOW 1/20  |  t=0.0s–3.0s  |  14:58:17
──────────────────────────────────────────────────────────────────
  Running VA analysis pipeline...

  ┌─ BEHAVIOR DISPATCH  [window 1  t≈0s] ────────────────────────┐
  │  Intent :  ENGAGE        (CHANGED)
  │  Pose   :  REACTION_POSE   speed=0.90×  dist=1.00×  dur=1.5s
  │  State  :  positive-high-arousal  (conf: 0.550)
  │  VA     :  valence=+0.353   arousal=+0.537
  └──────────────────────────────────────────────────────────────┘
[BITTLE-DEBUG] apply_reaction called, intent=ENGAGE, ra=ReactionAction(...)
[BITTLE] Executing intent: ENGAGE
[BITTLE] → s 4
[BITTLE] → khi

  Frames: 90  |  Analysis time: 10.64s
  ...

  ★  BEHAVIOR UPDATE:  —  →  ENGAGE  (first window)
```

The `[BITTLE-DEBUG]` line is a diagnostic print present in the current codebase.
It fires before any guard, confirming `apply_reaction()` was reached and showing the
intent value. Remove it once the integration is confirmed stable.

---

## Performance Tiers

| Mode | Frames/window | CPU time/window | Use case |
|---|---|---|---|
| `--sparse` (CPU) | 2 | ~5 s | Rapid logic + robot integration verification |
| Full (CPU) | 90 | ~40–50 s | Not recommended for real-time robot feedback |
| Full (GPU / Colab) | 90 | ~2–3 s | Final results, SIGDIAL data |

For fastest offline + robot results, run on **Google Colab** with GPU:

```bash
python run_offline.py samples/video.mp4 --device cuda --robot-port COM3
```

---

## Regenerating Calibration

If you update the model or run on a new dataset:

```bash
python pipeline/generate_calibration.py samples/ \
    --model_path models/jointcam_finetuned_v4.pt \
    --output config/calibration.json \
    --device cpu
```

This processes all videos in `samples/` and writes updated percentile statistics
(p10–p90, MAD, IQR, mean, std) to `config/calibration.json`. The pipeline uses
these to set scale-aware thresholds for volatility gating and state classification.

---

## Differences from the Online Pipeline

| Aspect | Offline | Online |
|---|---|---|
| Input source | Pre-recorded video file | Live webcam + microphone |
| Capture | Sequential video segment extraction | Continuous ring-buffer capture thread |
| Analysis | Sequential (one window at a time) | Pipelined (analysis runs parallel with capture) |
| Audio | Extracted from video container or separate file | Captured live via `sounddevice` |
| Momentum chaining | `last_v` / `last_a` passed explicitly | Stored inside `SpotReactionMapper` |
| Overflow | Not possible (sequential) | Warned if analysis exceeds window duration |
| Robot dispatch | Direct call after each window | `on_behavior_update` callback |
| Entry point | `run_offline.py` | `run_online.py` |

The stability gating (`ReactionHistory`), output format (BEHAVIOR DISPATCH + summary
table), and the full 9-stage per-window analysis stack are **identical** in both modes.
