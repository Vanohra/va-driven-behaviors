# Eval 2 — Confidence Gating Ablation

## What it measures

The `ReactionHistory.min_confidence` threshold in `online/online_session.py`
rejects windows where the pipeline's own confidence estimate is too low,
falling back to the current (stable) intent instead.  This eval asks:
**does gating at 0.55 meaningfully suppress low-quality windows without
over-suppressing valid state changes?**

Note: `pose_mode` / `va_pose_rate` are *not* evaluated here.  `pose_mode` is
determined by `SpotReactionMapper` upstream of `ReactionHistory` and cannot
be changed by ablating the confidence threshold.

## How to run

```bash
# 1. Generate session CSVs (one-time)
for f in samples/*.mp4; do python run_offline.py "$f"; done

# 2. Run eval (fast — replay-based, no model inference)
python evaluation/eval_2_confidence_gating/run_eval.py

# 3. Open notebook
jupyter lab evaluation/eval_2_confidence_gating/confidence_eval.ipynb
```

## Conditions

| Condition | `min_confidence` | `cooldown_s` |
|-----------|-----------------|-------------|
| `no_gating` | 0.0 | 0.5 |
| `with_gating` | 0.55 (production) | 0.5 |

Simulated time = `t_start` from each CSV row.

## results.csv columns

| Column | Description |
|--------|-------------|
| `video` | Source video stem |
| `condition` | `no_gating` or `with_gating` |
| `transition_rate` | n_changes / n_windows |
| `flip_rate` | A→B→A reversals / n_windows |
| `entropy` | Shannon entropy (bits) |
| `suppression_rate` | fraction of windows where effective_intent fell back to NEUTRAL due to low confidence |
| `mean_confidence_at_transition` | mean state_confidence at did_change=True windows in the original session CSV |
| `n_windows` | Total windows in the session |

## Known limitation

`state_confidence` is computed by the pipeline itself (a combination of
baseline stability score and trend confidence), not by a human annotator.
The suppression_rate measures pipeline self-consistency, not accuracy
relative to ground truth.
