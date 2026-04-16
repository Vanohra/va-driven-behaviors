# Eval 1 — ReactionHistory Cooldown Ablation

## What it measures

The pipeline-level cooldown in `ReactionHistory` (`online/online_session.py`)
prevents a single noisy window from triggering a behavior switch too soon after
the last change.  This eval asks: **does the cooldown reduce intent jitter
without suppressing genuine transitions?**

It does NOT evaluate the hardware-adapter cooldown (`INTENT_COOLDOWN_SEC` in
`bittle_adapter.py`) — that is Eval 4.

## How to run

```bash
# 1. Generate session CSVs (one-time, ~3 min per video)
for f in samples/*.mp4; do python run_offline.py "$f"; done

# 2. Run eval (fast — replay-based, no model inference)
python evaluation/eval_1_cooldown/run_eval.py

# 3. Open notebook
jupyter lab evaluation/eval_1_cooldown/cooldown_eval.ipynb
```

## Conditions

| Condition | `cooldown_s` | `min_confidence` |
|-----------|-------------|-----------------|
| `no_cooldown` | 0.0 | 0.55 |
| `with_cooldown` | 0.5 (production) | 0.55 |

Simulated time = `t_start` from each CSV row (window start in seconds).
This evaluates the 0.5 s cooldown against the realistic 3.0 s window spacing.

**Expected finding**: at 3.0 s window spacing >> 0.5 s cooldown, the cooldown
gate never fires (`n_suppressed_by_cooldown = 0`), confirming that the
production 0.5 s cooldown adds zero latency at the offline window rate. In
live-streaming scenarios where windows arrive faster, the cooldown would
meaningfully suppress jitter.

## results.csv columns

| Column | Description |
|--------|-------------|
| `video` | Source video stem |
| `condition` | `no_cooldown` or `with_cooldown` |
| `transition_rate` | n_changes / n_windows |
| `flip_rate` | A→B→A reversals / n_windows |
| `entropy` | Shannon entropy (bits) over intent distribution |
| `n_windows` | Total windows in the session |
| `n_suppressed_by_cooldown` | Windows blocked by the cooldown gate |
| `intent_counts` | JSON string — {intent: count} |

## Known limitation

This eval measures pipeline stability (transition / flip rates), not whether
the chosen behavior is *appropriate* for the situation. Appropriateness
requires a user study.
