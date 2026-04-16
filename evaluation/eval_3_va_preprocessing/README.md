# Eval 3 — Raw vs. Preprocessed VA Signal Quality

## What it measures

`emotion_analyzer.preprocess_series()` applies Winsorization, a Hampel filter,
and rolling-median smoothing to the raw per-frame valence/arousal series before
any downstream analysis.  This eval asks: **does preprocessing measurably reduce
signal noise and jitter compared to the raw frame-by-frame output?**

Metrics compared (per video, per channel, per condition):

| Metric | What it captures |
|--------|-----------------|
| MAD | Median absolute deviation — robust spread |
| IQR | Interquartile range — spread without tails |
| frame_delta_variance | `var(diff(series))` — per-frame jitter |
| outlier_rate | `(n_winsorized + n_hampel_replaced) / (2 × n_frames)` — fraction of frames corrected (0 for raw) |

## How to run

```bash
# Smoke-test on 1 video first (~25 s)
python evaluation/eval_3_va_preprocessing/run_eval.py --sample 1

# Full run (30 videos, ~12–15 min)
python evaluation/eval_3_va_preprocessing/run_eval.py

# Open notebook
jupyter lab evaluation/eval_3_va_preprocessing/va_preprocessing_eval.ipynb
```

## Important notes

- **Slow**: re-runs the JointCAM vision model on every video.
  `--sample N` processes only the first N files (alphabetical order).
- Saves `eval_3_sample_series.npz` alongside `results.csv` for the
  time-series visualisation in the notebook (requires one successful run).
- The preprocessing configuration used is identical to the one inside
  `analyze_emotion_stream` (winsor_limits=(0.01,0.01), hampel_window=7,
  smooth_window=5, smooth_method='rolling_median').

## results.csv columns

| Column | Description |
|--------|-------------|
| `video` | Source video stem |
| `condition` | `raw` or `processed` |
| `channel` | `valence` or `arousal` |
| `mad` | Median absolute deviation of the series |
| `iqr` | Interquartile range |
| `frame_delta_variance` | Variance of frame-to-frame differences (jitter) |
| `outlier_rate` | Fraction of frames corrected by preprocessing (0 for raw) |
| `n_frames` | Number of frames in the series |

## Known limitation

This eval measures signal *cleanliness* (noise reduction), not whether the
preprocessed signal is closer to the subject's true emotional state.  Validating
the latter requires ground-truth annotations, which await the user study
(IRB pending).  This eval is framed as pipeline validation, not appropriateness
evaluation.
