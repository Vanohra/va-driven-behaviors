# Online Pipeline — Detailed Reference

`run_online.py` runs a live valence-arousal session from a webcam and microphone.
It uses a **ring buffer + pipelined architecture**: the camera never pauses, and
analysis runs in a worker thread in parallel with ongoing capture.

---

## Architecture

### Three Concurrent Loops

```
┌──────────────────────────────────────────────────────────────────┐
│ Loop A — CaptureThread (runs the full session, never pauses)     │
│   cap.read() → resize if needed → ring_buffer.push_frame()       │
│                                                                   │
│ Loop B — AudioThread (runs the full session, optional)           │
│   sd.InputStream callback → ring_buffer.push_audio()             │
│                                                                   │
│ Loop C — Main thread (analysis timer, polls every 20 ms)         │
│   every window_dur seconds:                                       │
│     snapshot = ring_buffer.snapshot()   ← atomic copy under lock │
│     executor.submit(analysis_worker, snapshot)                    │
│   after session_dur: executor.shutdown(wait=True)                 │
└──────────────────────────────────────────────────────────────────┘
```

### Ring Buffer

The `RingBuffer` holds the last `window_dur × 1.5` seconds of data at all times:

- **Video frames** — `deque(maxlen = fps × window_dur × 1.5)`. At 15 fps / 3 s
  windows that is 67 slots. When full, the oldest frame falls off automatically.
- **Audio chunks** — trimmed to `sample_rate × window_dur × 1.5` total samples.
  Old chunks are removed from the front when the budget is exceeded.

The 1.5× factor absorbs timer jitter and ensures no gap between snapshot boundaries.

`snapshot()` copies both structures under a single lock so that the frames and
audio returned always correspond to the same instant in time. The worker thread
receives a frozen copy and never touches the live buffer.

### Worker Pool

`ThreadPoolExecutor(max_workers=2)` — one active worker plus one queued slot for
overflow. If a previous worker is still running when the next snapshot fires,
a warning is printed but the new worker is still submitted (no data is lost).

Analysis timing is printed for every window. If analysis time exceeds `window_dur`,
an `!` flag appears in the session summary table.

---

## Per-Window Pipeline (9 stages, identical to offline)

Each worker receives a frozen snapshot and processes it through the same 9-stage
stack as the offline pipeline:

1. **Save snapshot** — frames written to a temp `.avi` (XVID codec); audio
   converted from float32 → int16 and written to a temp `.wav` at 16 kHz.
2. **Feature extraction** — JointCAM visual and audio encoders.
3. **Alignment** — resample shorter modality to match the longer.
4. **Inference** — `valence_series[T]` and `arousal_series[T]` from JointCAM.
5. **Robust preprocessing** — winsorize → Hampel filter → rolling-median smoothing.
6. **Session baseline** — trimmed-mean scalar for valence and arousal.
7. **Trend analysis** — Theil-Sen slope; direction and delta.
8. **Volatility** — MAD; `volatility = max(valence_vol, arousal_vol)`.
9. **Reaction mapping** — volatility-gated 5-intent policy → `ReactionAction`.

### Stability Gating (`ReactionHistory`)

After analysis, the proposed intent passes through two gates before being accepted:

- **Confidence gate** — reject if `state_confidence < --min-confidence` (default 0.55).
- **Cooldown gate** — reject if fewer than `--cooldown` seconds since the last
  accepted change (default 0.5 s).

The **effective intent** is dispatched to the robot via the `on_behavior_update`
callback. `ra.intent` is overridden with `effective_intent` before calling
`robot.apply_reaction(ra)`, so the robot always acts on the filtered result.

---

## Running the Online Pipeline

### Without robot (terminal output only)

```bash
python run_online.py
```

### With Bittle X — hardware on serial port

```bash
python run_online.py --robot-port COM3
```

### With Bittle X — mock mode (no hardware required)

```bash
python run_online.py --mock-robot
```

### Common invocations

```bash
# Default: 30s session, 3s windows, CPU, no robot
python run_online.py

# With robot on COM3
python run_online.py --robot-port COM3

# Mock robot — verify intent dispatch without hardware
python run_online.py --mock-robot

# GPU — 3s windows are achievable, robot on COM3
python run_online.py --device cuda --robot-port COM3

# Adjust for CPU — widen window so analysis fits within the window period
python run_online.py --duration 180 --window 60 --robot-port COM3

# Disable audio to reduce pipeline work
python run_online.py --window 60 --no-audio --robot-port COM3

# Sparse sampling — very fast, good for verifying robot integration
python run_online.py --sparse --mock-robot

# Second webcam
python run_online.py --camera 1 --robot-port COM3

# Strict stability gates
python run_online.py --min-confidence 0.6 --cooldown 2 --robot-port COM3

# Keep temp files for debugging snapshots
python run_online.py --no-cleanup --debug --mock-robot
```

---

## Full Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--duration S` | `30.0` | Total session duration in seconds |
| `--window S` | `3.0` | Analysis window size in seconds |
| `--camera N` | `0` | OpenCV camera index |
| `--fps F` | `15.0` | Webcam capture frame rate |
| `--no-audio` | off | Disable microphone capture (video-only mode) |
| `--device cpu\|cuda` | `cpu` | Torch inference device |
| `--model PATH` | auto | Path to `.pt` model file |
| `--min-confidence C` | `0.55` | Minimum confidence to accept an intent change |
| `--cooldown S` | `0.5` | Minimum seconds between behavior changes |
| `--sparse` | off | Process only 2 frames per window — fast CPU testing |
| `--no-cleanup` | off | Keep temp snapshot files after the session |
| `--debug` | off | Verbose per-stage output |
| `--robot-port PORT` | off | Serial port for Petoi Bittle X (e.g. `COM3`, `/dev/ttyUSB0`) |
| `--mock-robot` | off | Enable mock mode: print commands instead of sending to hardware |

**Notes:**
- If neither `--robot-port` nor `--mock-robot` is passed, `robot = None` and no
  `[BITTLE]` output will appear — the pipeline runs in terminal-only mode.
- `--mock-robot` without `--robot-port` uses `COM7` in log messages but never opens it.
- On CPU, analysis typically takes 40–50 s per 3 s window. Use `--window 60` to avoid
  worker overflow, or use GPU.

---

## Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║                      VA STREAMING SESSION                      ║
║                   (Ring Buffer + Pipelined)                    ║
╚════════════════════════════════════════════════════════════════╝
  Started:      2026-04-15 14:22:01
  Duration:     30s  |  10 windows × 3s each
  Mode:         Camera never pauses — analysis runs in parallel
  Overflow:     warned if analysis takes longer than 3s
  Confidence:   0.55  |  Cooldown: 0.5s

[BITTLE] Connected on COM3 @ 115200 baud.
[BITTLE] → kbalance
  Robot:    Petoi Bittle X (hardware on COM3)

Camera opened: 640×360 @ 30.0 fps  (requested 320×240 @ 15.0)
  Camera running.  First analysis snapshot in 3s.

──────────────────────────────────────────────────────────────────
  WINDOW 1/10  |  t≈3s  |  14:22:04  |  67 frames in buffer
──────────────────────────────────────────────────────────────────
  [Window 1] Running VA analysis pipeline...

  ┌─ BEHAVIOR DISPATCH  [window 1  t≈3s] ────────────────────────┐
  │  Intent :  ENGAGE        (CHANGED)
  │  Pose   :  VA_POSE         speed=0.90×  dist=1.00×  dur=1.5s
  │  State  :  positive-high-arousal  (conf: 0.712)
  │  VA     :  valence=+0.341   arousal=+0.478
  └──────────────────────────────────────────────────────────────┘
[BITTLE-DEBUG] apply_reaction called, intent=ENGAGE, ra=ReactionAction(...)
[BITTLE] Executing intent: ENGAGE
[BITTLE] → s 4
[BITTLE] → khi
```

---

## Analysis Latency and Overflow

The pipeline must run JointCAM inference on each snapshot. On CPU with a 3 s window
(~67 frames), inference typically takes **40–50 seconds** — far longer than the 3 s
window period. This means:

- Worker 1 starts at t=3 s and finishes at t≈43 s — the decision arrives 40 s late.
- Worker 2 starts at t=6 s while Worker 1 is still running — overflow warning fires.
- The executor queues Worker 2 behind Worker 1 (max_workers=2).

**Recommended window sizes by hardware:**

| Hardware | Recommended `--window` | Expected analysis time |
|---|---|---|
| CPU only | `--window 60` | ~45–55 s |
| CPU + `--sparse` | `--window 10` | ~4–6 s |
| GPU (CUDA) | `--window 3` | ~2–3 s |
| GPU + `--no-audio` | `--window 3` | ~1–2 s |

**CPU workaround commands:**

```bash
# 3 × 60s windows — analysis fits comfortably on CPU
python run_online.py --duration 180 --window 60

# Sparse sampling — 2 frames per window, very fast
python run_online.py --sparse

# Disable audio to reduce pipeline work
python run_online.py --window 60 --no-audio
```

---

## Hardware Integration — Bittle X

The robot is connected via `--robot-port` and driven through the `on_behavior_update`
callback that `StreamingSession` calls immediately when each worker reaches a decision.

The callback receives `(intent: str, reaction_action, analysis: dict)` where `intent`
is already the **stability-gated** effective intent (not the raw model output).
`reaction_action.intent` is overridden with this value before `apply_reaction()` is
called, so the robot always acts on the filtered, confidence-checked intent.

### Robot dispatch flow (online mode)

```
StreamingSession._analysis_worker()
    └── ReactionHistory.evaluate()           ← produces effective_intent
    └── _dispatch_behavior()
        └── on_behavior_update(intent, ra, analysis)
            └── ra.intent = intent           ← override with stability-gated intent
            └── robot.apply_reaction(ra)
                └── BittleXAdapter._dispatch_or_queue()
                    └── _run_intent()        ← background thread
                        └── _execute_intent()
```

### Using the callback in your own code

```python
from online.streaming_session import StreamingSession
from online.window_analyzer import WindowAnalyzer
from robot.bittle_adapter import BittleXAdapter

robot = BittleXAdapter(port="COM3")

def on_behavior_update(intent: str, reaction_action, analysis: dict):
    # Called immediately when each worker reaches a decision.
    # intent is the stability-gated effective intent.
    if reaction_action is not None:
        reaction_action.intent = intent  # always override with gated intent
        robot.apply_reaction(reaction_action)

session = StreamingSession(
    window_analyzer=analyzer,
    on_behavior_update=on_behavior_update,
    cooldown_s=0.5,
    min_confidence=0.55,
)
session.run()
robot.disconnect()
```

See [robot/bittle.md](../robot/bittle.md) for full adapter documentation, tuning
constants, and the debugging checklist.

---

## Differences from the Offline Pipeline

| Aspect | Online | Offline |
|---|---|---|
| Input source | Live webcam + microphone | Pre-recorded video file |
| Capture | Continuous ring-buffer thread | Sequential segment extraction |
| Analysis | Pipelined (parallel with capture) | Sequential (one window at a time) |
| Audio | Live `sounddevice` stream | Extracted from video container |
| Overflow | Possible; warned; queued | Not possible |
| Robot dispatch | `on_behavior_update` callback | Direct call after each window |
| Entry point | `run_online.py` | `run_offline.py` |

The stability gating (`ReactionHistory`), output format (BEHAVIOR DISPATCH + summary
table), and the full 9-stage per-window analysis stack are **identical** in both modes.
