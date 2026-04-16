# Eval 4 — Adapter Cooldown Effect on Robot Command Rate

## What it measures

`BittleXAdapter._execute_intent` in `robot/bittle_adapter.py` guards against
servo thrashing with two mechanisms:

1. **Deduplication** — skip if `intent == current_pose`
2. **Rate limiting** — skip if `now - last_intent_time < INTENT_COOLDOWN_SEC` (2.5 s)

This eval asks: **how much does `INTENT_COOLDOWN_SEC=2.5` reduce the serial
command rate, and why does it matter more in offline (fast-replay) mode than
in online (real-time) mode?**

This is distinct from Eval 1 (pipeline intent stability): a pipeline that
emits *stable* intents can still thrash the robot if it dispatches them faster
than the hardware can execute them.

## How to run

```bash
# 1. Generate session CSVs (one-time)
for f in samples/*.mp4; do python run_offline.py "$f"; done

# 2. Run eval (fast — pure simulation, no hardware, no model inference)
python evaluation/eval_4_adapter_cooldown/run_eval.py

# 3. Open notebook
jupyter lab evaluation/eval_4_adapter_cooldown/adapter_cooldown_eval.ipynb
```

## Conditions × timing modes

| Condition | `INTENT_COOLDOWN_SEC` | `timing_mode` | Window interval |
|-----------|----------------------|---------------|----------------|
| `no_cooldown` | 0.0 | `online_mode` | 3.0 s |
| `no_cooldown` | 0.0 | `offline_mode` | 0.5 s |
| `with_cooldown` | 2.5 (production) | `online_mode` | 3.0 s |
| `with_cooldown` | 2.5 (production) | `offline_mode` | 0.5 s |

The simulation uses `simulated_time` incremented by the window interval.
`time.sleep` is **never** called.

## results.csv columns

| Column | Description |
|--------|-------------|
| `video` | Source video stem |
| `condition` | `no_cooldown` or `with_cooldown` |
| `timing_mode` | `online_mode` (3 s/window) or `offline_mode` (0.5 s/window) |
| `n_commands_sent` | Intents that passed both dedup and cooldown checks |
| `n_commands_dropped` | Total blocked intents (dedup + cooldown) |
| `command_rate` | Commands per minute of simulated time |
| `dedup_drop_rate` | Dedup-blocked fraction of windows |
| `cooldown_drop_rate` | Cooldown-blocked fraction of windows |
| `n_windows` | Total windows |

## Design note — why 2.5 s?

The ENGAGE motion (`khi` wave) has a motion budget of 3.5 s in
`_INTENT_COMMANDS`.  A cooldown of 2.5 s ensures at most one command per
physical motion cycle — the robot will not receive a new command while still
mid-wave.  The cooldown is shorter than the ENGAGE budget by design, because
faster intents (NEUTRAL, CHECK_IN) complete in ≤ 2.0 s.

## Known limitation

Timing is simulated, not measured on hardware.  Real serial latency,
ESP32 response time, and firmware queuing may cause additional effective
cooldown beyond what this simulation models.
