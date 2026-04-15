# Bittle X Adapter — Complete Reference

`bittle_adapter.py` is the bridge between the VA pipeline and the physical
Petoi Bittle X robot dog. It receives behavior intents from the pipeline,
translates them into Petoi serial commands, and manages execution timing so
the robot always looks deliberate and expressive rather than jittery.

---

## Quick Start

### Run offline pipeline with hardware

```bash
python run_offline.py samples/video67.mp4 --robot-port COM3
```

### Run offline pipeline in mock mode (no hardware)

```bash
python run_offline.py samples/video67.mp4 --mock-robot
```

### Run online (live) pipeline with hardware

```bash
python run_online.py --robot-port COM3
```

### Run online pipeline in mock mode

```bash
python run_online.py --mock-robot
```

**Finding your port:**
- Windows: open Device Manager → Ports (COM & LPT). The Bittle shows as a
  CP210x or CH340 device, typically `COM3`–`COM9`.
- Linux: `ls /dev/ttyUSB*` or `ls /dev/ttyACM*` before and after plugging in.
- macOS: `ls /dev/cu.usbserial-*`.

If the port shows `Access is denied`, close Arduino IDE or any other app that
has the port open.

---

## Robot flags (both pipelines)

| Flag | Description |
|---|---|
| `--robot-port PORT` | Open this serial port (e.g. `COM3`, `/dev/ttyUSB0`). If omitted, `robot = None` and no robot commands are sent. |
| `--mock-robot` | Mock mode: print all serial commands to terminal with `[BITTLE-MOCK]` prefix. No hardware required. |

If neither flag is passed, the robot adapter is never instantiated and
`apply_reaction()` is never called — no `[BITTLE]` output will appear in
the terminal at all. This is the most common cause of a "silent" robot.

---

## The five intents and what the robot does

The pipeline produces one of five intents per window. Each maps to a specific
physical behavior.

| Intent | Command(s) | What it looks like | Motion budget |
|---|---|---|---|
| `NEUTRAL` | `kbalance` | Balanced, relaxed standing pose | 1.5 s |
| `CAUTION` | `kbalance` | Same standing pose, signals alertness contextually | 1.5 s |
| `CHECK_IN` | `knd` | Nods — a gentle acknowledgement gesture | 2.0 s |
| `ENGAGE` | `khi` | Waves — a full, expressive greeting | 3.5 s |
| `DE_ESCALATE` | `kbk` → `ksit` | Backs up for 2 s, then sits down | 2.0 s + 1.5 s |

The **motion budget** controls how long the adapter treats the robot as "busy"
after sending the command. It should match how long the physical motion visibly
takes on your hardware. Tune in `_INTENT_COMMANDS` if needed.

---

## Architecture overview

Four layers sit between the pipeline and the robot. Each has one responsibility.

```
VA pipeline
    ↓  ReactionAction(intent=...)
ReactionHistory stability gate        ← upstream, in online_session.py
    ↓  effective_intent (overrides ra.intent before calling apply_reaction)
apply_reaction()                      ← entry point, Layer 1
    ↓
_dispatch_or_queue()                  ← single-slot queue, Layer 2
    ↓
_run_intent()                         ← background thread launcher, Layer 3
    ↓
_execute_intent()                     ← guards + serial dispatch, Layer 4
    ↓
_send_smooth() → send_command()       ← speed prefix + raw serial I/O
    ↓
Bittle X robot (ESP32 over serial)
```

---

## Layer 1 — `apply_reaction()` (entry point)

```python
def apply_reaction(self, reaction_action) -> None:
```

The only public method the pipeline calls. Both `run_offline.py` and
`run_online.py` call it after every stability-gated behavior dispatch.

It does two things:
1. Guards against a `None` reaction (returns silently if so).
2. Extracts the `intent` string from the `ReactionAction` dataclass and passes
   it to `_dispatch_or_queue`.

**Important:** the pipeline always overrides `ra.intent` with `effective_intent`
(the stability-gated output) before calling this method. The raw model intent
is never sent to the robot directly.

**Debug print:** the current codebase includes a diagnostic at the very start
of this method:

```
[BITTLE-DEBUG] apply_reaction called, intent=ENGAGE, ra=ReactionAction(...)
```

This fires before any guard, confirming the function was reached and showing
the intent value. Remove it once integration is confirmed stable.

`apply_reaction_pose` is kept as an alias for backwards compatibility.

---

## Layer 2 — `_dispatch_or_queue()` (single-slot queue)

```python
def _dispatch_or_queue(self, intent: str) -> None:
```

Solves the core timing problem: the offline pipeline processes video faster
than real time, so intents arrive faster than the robot can execute them.
Without a queue, the robot would race through expressions in rapid succession.

**The rules:**

1. Robot is free (`_executing = False`) → dispatch immediately via `_run_intent`.
2. Robot is busy (`_executing = True`) → store intent in `_pending_intent` (the
   single waiting slot).
3. New intent arrives while slot is occupied → **replace** the waiting intent
   with the newer one and log a `[BITTLE] Replacing stale pending ...` message.

The replacement rule is the key research decision. The robot always reacts to
the most recent emotional content rather than replaying a stale backlog. If 10
windows of ENGAGE arrive while the robot is waving, it waves once and picks up
whatever the final state was — it does not wave 10 times.

**What happens with repeated identical intents (e.g. 6 windows of CHECK_IN):**

```
Window 1 → robot free → fires knd, _executing = True
Window 2 → robot busy → _pending_intent = "CHECK_IN"
Window 3 → robot still busy → replaces pending (same intent, silent)
knd finishes → picks up pending → calls _execute_intent("CHECK_IN")
_execute_intent: current_pose already "CHECK_IN" → dedup guard returns
Robot goes idle. Windows 4–6 repeat the same pattern.
```

Repeated identical intents collapse correctly — the motion executes once, then
the robot holds the pose.

---

## Layer 3 — `_run_intent()` (background thread)

```python
def _run_intent(self, intent: str) -> None:
```

`_execute_intent` uses `time.sleep` to hold the robot busy for the full motion
budget. If that ran on the pipeline's main thread, it would block the entire
pipeline for 1.5–3.5 seconds per window.

`_run_intent` spawns a **daemon thread** to run `_execute_intent` in the
background. The pipeline is free to receive and queue new intents while the
robot is physically moving.

`_executing` is set to `True` synchronously before the thread starts — there is
no window where a second `apply_reaction` call could slip through and think the
robot is free.

**On finish, the worker thread:**
1. Sets `_executing = False`.
2. Atomically reads and clears `_pending_intent` in a single step.
3. If a pending intent was waiting, calls `_run_intent` recursively with it.

The clear-before-recurse order prevents a race: a new `apply_reaction` arriving
at the exact moment of completion cannot re-read a slot that is already being
consumed.

**Exception handling:** exceptions inside the worker are currently caught by the
`try/finally` block. The `finally` always resets `_executing` so the queue is
never permanently stuck. However, exceptions are not currently logged. If the
robot stops responding mid-session with no terminal output:

```python
# In _worker(), add an except clause inside the try:
try:
    self._execute_intent(intent)
except Exception as e:
    print(f"[BITTLE] Worker exception for '{intent}': {e}")
finally:
    ...
```

This will surface silent crashes that `try/finally` alone hides.

---

## Layer 4 — `_execute_intent()` (guards + serial dispatch)

```python
def _execute_intent(self, intent: str) -> None:
```

Converts intents to serial commands. Three guards run before any command is sent.

### Guard 1 — Deduplication

```python
if intent == self.current_pose:
    return
```

If the robot is already in this state, the call is a no-op. Prevents redundant
serial writes for sustained states — if ENGAGE holds for several windows, `khi`
is only sent once.

### Guard 2 — Rate-limiting cooldown

```python
if now - self._last_intent_time < INTENT_COOLDOWN_SEC:   # default 2.5 s
    print(f"[BITTLE] Cooldown active — skipping intent '{intent}'")
    return
```

Even after the queue releases a pending intent, this guard drops it if fewer
than 2.5 s have passed since the last executed command. This absorbs any
residual burst that gets past the queue and protects the servos from
mid-motion thrashing.

**Known interaction with the queue:** if a queued intent is picked up but the
2.5 s cooldown has not yet expired, `_execute_intent` drops it silently and
`_executing` flips back to `False` with nothing happening. The robot goes idle.
If you see occasional skipped reactions after a queue pickup, lower
`INTENT_COOLDOWN_SEC` or increase the motion budgets so the cooldown has
expired by the time the queue releases.

### Guard 3 — Settle delay

After sending a single-step command, the thread sleeps for the motion budget:

```python
time.sleep(settle)
```

This is what keeps `_executing = True` for the full physical duration of the
motion. Without it, `_execute_intent` would return immediately after the serial
write, `_executing` would flip to `False`, and the queue would release the next
intent before the robot had finished moving.

### Multi-step sequences (DE_ESCALATE)

`DE_ESCALATE` has two steps: `kbk` (back up, 2.0 s) → `ksit` (sit, 1.5 s).
The first command is sent immediately; the rest are fired by `_schedule_sequence`
via a `threading.Timer` after the first command's delay.

**Known limitation:** `_schedule_sequence` uses a Timer (non-blocking), so
`_execute_intent` returns immediately after sending `kbk`. This means
`_executing` flips to `False` before `ksit` fires, and the queue can release a
new intent mid-sequence.

**Fix if needed:** replace `_schedule_sequence` with sequential `time.sleep`
calls directly inside `_execute_intent`. This is safe because `_execute_intent`
already runs in a background thread:

```python
# Replace the else branch in _execute_intent with:
else:
    for cmd, delay in commands:
        self._send_smooth(cmd)
        if delay > 0:
            time.sleep(delay)
```

---

## Movement speed — `_send_smooth()` and `SERVO_SPEED`

```python
SERVO_SPEED = 4   # module-level constant (0 = slowest, 10 = fastest)

def _send_smooth(self, cmd: str) -> None:
    self.send_command(f"s {SERVO_SPEED}")
    time.sleep(0.05)
    self.send_command(cmd)
```

The Petoi firmware accepts a speed control command `s` that sets how fast servos
transition between positions. Lower values produce slower, smoother motions.
The firmware default is approximately 8, which feels abrupt for expressive
research contexts.

`_send_smooth` is called for every expressive skill command (`kbalance`, `knd`,
`khi`, `kbk`, `ksit`). It is **not** used for utility or safety commands:
- `p` (pause/stop) — must respond immediately
- `d` (rest/sleep in `disconnect()`) — must respond immediately
- `kbalance` in `_connect()` — startup command, must respond immediately

### Tuning SERVO_SPEED

| Value | Character | Good for |
|---|---|---|
| 2–3 | Very slow, deliberate | Calm or therapeutic contexts |
| **4–5** | Smooth and readable — **recommended default** | Research demos |
| 6–7 | Close to natural speed | Responsive real-time interaction |
| 8–10 | Firmware default, abrupt | Not recommended for VA research |

### Firmware compatibility note

Two command formats exist across Petoi firmware versions:
- `"s 4"` (with space) — used in current code, works on most recent firmware
- `"s4"` (no space) — required by some older firmware versions

If `SERVO_SPEED` changes have no visible effect on the robot, try removing the
space: change `f"s {SERVO_SPEED}"` to `f"s{SERVO_SPEED}"` in `_send_smooth`.

---

## Serial I/O — `send_command()`

```python
def send_command(self, cmd: str) -> None:
```

The lowest-level method. Writes a single command string to the serial port,
appending `\n` as a terminator. In mock mode, prints `[BITTLE-MOCK] → {cmd}`
instead of writing to the port. Catches and prints serial write errors without
raising — the pipeline always continues.

**Note:** every command sent to the robot is logged to the terminal. In mock
mode the prefix is `[BITTLE-MOCK] →`. On real hardware the prefix is `[BITTLE] →`.
You can use this log to verify the exact command sequence the robot receives.

---

## Module-level tuning constants

| Constant | Default | What it controls |
|---|---|---|
| `INTENT_COOLDOWN_SEC` | `2.5` | Minimum seconds between distinct intent executions. Lower this if genuine transitions are being dropped after queue pickups. |
| `SERVO_SPEED` | `4` | Servo transition speed (0 = slowest, 10 = fastest). Increase for faster/snappier motion; decrease for smoother/calmer motion. |

Motion budgets (how long the robot stays "busy") are tuned per-intent in
`_INTENT_COMMANDS`. Edit the second element of each tuple:

```python
_INTENT_COMMANDS = {
    "NEUTRAL":     [("kbalance", 1.5)],   # ← tune this value
    "CAUTION":     [("kbalance", 1.5)],
    "CHECK_IN":    [("knd",      2.0)],
    "ENGAGE":      [("khi",      3.5)],
    "DE_ESCALATE": [("kbk",      2.0), ("ksit", 1.5)],
}
```

---

## Connection lifecycle

### `_connect()`

Called from `__init__`. Opens the serial port, waits **2 seconds** for the
ESP32 to reset after the DTR toggle (this wait is required — commands sent
immediately after `open()` are silently ignored by the firmware), then sends
`kbalance` to put the robot in a ready standing pose.

If the port is inaccessible, the adapter automatically falls back to mock mode
with a diagnostic message. The `"Access is denied"` tip suggests closing
Arduino IDE, which is the most common cause on Windows.

### `disconnect()`

Sends `d` (rest/sleep — relaxes all servos and saves battery), then closes the
serial port. Should always be called at session end via `robot.disconnect()`.
Both `run_offline.py` and `run_online.py` call this in their `finally` blocks.

---

## Graceful degradation

The adapter never crashes the pipeline. Every failure mode degrades gracefully:

| Failure | Behaviour |
|---|---|
| `pyserial` not installed | Mock mode, warning logged at import time |
| Serial port unavailable | Mock mode, diagnostic printed in `_connect()` |
| Port access denied | Mock mode, "close Arduino IDE" tip printed |
| Serial write error | Error printed, execution continues |
| Unknown intent string | Defaults to `kbalance`, warning printed |
| Worker thread exception | `_executing` reset via `finally`, queue unblocked |

---

## Integration with the pipeline

### `run_offline.py` dispatch block

```python
if robot is not None and analysis is not None:
    ra = analysis.get("reaction_action")
    if ra is not None:
        ra.intent = effective     # override with stability-gated intent
        robot.apply_reaction(ra)
```

`effective` is the output of `ReactionHistory.evaluate()` — the confidence-
checked and cooldown-gated intent. The raw model intent in `ra.intent` is
discarded.

### `run_online.py` callback

```python
def on_behavior_update(intent, ra, analysis):
    if ra is not None:
        ra.intent = intent        # override with stability-gated intent
        robot.apply_reaction(ra)
```

`intent` arrives as `effective_intent` from `StreamingSession._dispatch_behavior`,
so the same stability guarantee holds.

---

## Debugging checklist

| Symptom | Likely cause and fix |
|---|---|
| No `[BITTLE]` output at all (not even connection messages) | `robot = None` — missing `--robot-port` or `--mock-robot` flag |
| Connection message appears but no per-window output | `apply_reaction()` not being called — check `[BITTLE-DEBUG]` print; if absent, robot is None during session loop |
| `[BITTLE-DEBUG]` shows `intent=None` | `ra.intent` override not applied upstream — check `run_offline.py` line 413 or `run_online.py` `on_behavior_update` |
| Robot executes window 1 then goes silent | Cooldown blocking queue pickups — lower `INTENT_COOLDOWN_SEC` (currently 2.5 s) or verify motion budgets are long enough |
| Robot executes same intent twice rapidly | Dedup guard not firing — `current_pose` may not be set; check that `_execute_intent` is running (not silently dropped by cooldown) |
| `DE_ESCALATE` interrupted mid-sequence | Queue releasing during `_schedule_sequence` Timer delay — see known limitation above |
| Speed control has no visible effect | Try `f"s{SERVO_SPEED}"` (no space) in `_send_smooth` for older firmware |
| Robot jitters or moves too fast | `SERVO_SPEED` too high — lower to 3–4; also verify motion budgets match physical durations |
| `[BITTLE] Cooldown active` printed every window | `INTENT_COOLDOWN_SEC` (2.5 s) longer than inter-window period — lower it, or increase motion budgets so cooldown has expired by the time the next intent arrives |
| Worker thread silently crashing | Add `except Exception as e: print(...)` inside `_worker()` — see Layer 3 note above |
