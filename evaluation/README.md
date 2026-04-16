# VA Pipeline — Evaluation Suite

## Strategy

Because an IRB-approved user study is pending, this suite uses **controlled ablations** over 30 pre-recorded videos in `samples/`. Each evaluation isolates one pipeline component and measures its effect by replaying existing session CSVs (or re-running only the vision model, for Eval 3). No human ground-truth labels for behavioral appropriateness are used.

This approach is appropriate for a systems paper: it validates that each component does what it claims mechanically (reduces jitter, suppresses low-confidence windows, cleans the signal, limits servo thrash) before any human-subjects work begins.

---

## Evaluation Overview

| # | Name | Target component | Robot needed? | Reruns model? |
|---|------|-----------------|---------------|---------------|
| 1 | Cooldown ablation | `ReactionHistory.cooldown_s` | No | No |
| 2 | Confidence gating ablation | `ReactionHistory.min_confidence` | No | No |
| 3 | Raw vs preprocessed VA signal | `emotion_analyzer.preprocess_series` | No | **Yes** (~12–15 min) |
| 4 | Adapter cooldown effect | `BittleXAdapter.INTENT_COOLDOWN_SEC` | Simulated | No |

Evals 1, 2, and 4 are fast (replay-based, seconds to run).
Eval 3 is slow (~25 s/video × 30 videos).

---

## Shared Color Palette

All notebooks import `INTENT_COLORS` from `eval_utils.py`. Never hardcode these per-notebook.

| Intent | Hex |
|--------|-----|
| NEUTRAL | `#B4B2A9` |
| CHECK_IN | `#5DCAA5` |
| ENGAGE | `#7F77DD` |
| DE_ESCALATE | `#D85A30` |
| CAUTION | `#EF9F27` |

---

## Setup

### 1 — Generate all 30 session CSVs

From the project root:

```bash
for f in samples/*.mp4; do python run_offline.py "$f"; done
```

Each run appends a CSV to `evaluation/session_csvs/`.  
Expected: `video29_results.csv`, `video30_results.csv`, …, `video34_results.csv` (and any others in `samples/`).

### 2 — Run all evaluations

```bash
python evaluation/eval_1_cooldown/run_eval.py
python evaluation/eval_2_confidence_gating/run_eval.py
python evaluation/eval_3_va_preprocessing/run_eval.py          # slow; use --sample 1 to smoke-test
python evaluation/eval_4_adapter_cooldown/run_eval.py
```

Each script writes `results.csv` into its own directory (overwrites on re-run).

### 3 — Open notebooks

Launch Jupyter from the project root, then open any notebook. Each notebook assumes `results.csv` already exists in the same directory.

---

## Production Parameter Reference

These are the values used when generating session CSVs with `run_offline.py`:

| Parameter | Value | Source |
|-----------|-------|--------|
| `--window` | 3.0 s | `run_offline.py` line ~79 |
| `--min-confidence` | 0.55 | `run_offline.py` line ~107 |
| `--cooldown` | 0.5 s | `run_offline.py` line ~113 (offline pipeline) |
| `OnlineSession.cooldown_s` | 8.0 s | `online_session.py` — live mode only, **not used in evals** |
| `INTENT_COOLDOWN_SEC` | 2.5 s | `bittle_adapter.py` — hardware adapter |

---

## Known Limitations (for the paper)

- **N = 30 videos**: results have reasonable statistical power but no human ground-truth labels for behavioral appropriateness.
- **Pipeline-internal confidence**: `state_confidence` is computed by the pipeline itself, not labeled by a human annotator. Evals 1–2 measure pipeline self-consistency, not ground-truth accuracy.
- **Evals 1–3 measure pipeline properties**, not behavioral appropriateness. Whether a chosen intent is correct for the situation is a question for the user study.
- **Eval 4 uses simulated timing**, not real hardware measurements. Actual servo latency may differ from the motion-budget constants in `_INTENT_COMMANDS`.
