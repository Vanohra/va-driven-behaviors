"""
Shared utilities for the VA pipeline evaluation suite.

Import this module at the top of every eval notebook and run_eval.py:

    from eval_utils import INTENT_COLORS, load_session_csv, load_all_sessions, \
                           compute_metrics, save_results
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ── Shared intent color palette ───────────────────────────────────────────────
# Use this in ALL notebooks — never hardcode colors per notebook.

INTENT_COLORS: Dict[str, str] = {
    "NEUTRAL":     "#B4B2A9",
    "CHECK_IN":    "#5DCAA5",
    "ENGAGE":      "#7F77DD",
    "DE_ESCALATE": "#D85A30",
    "CAUTION":     "#EF9F27",
}


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_session_csv(path) -> pd.DataFrame:
    """
    Load one session CSV produced by run_offline.py and enforce correct dtypes.

    Expected columns (from _save_session_csv):
        window_index, t_start, t_end, proposed_intent, effective_intent,
        did_change, change_reason, pose_mode, state_label, state_confidence,
        valence, arousal, valence_trend, arousal_trend,
        valence_volatility, arousal_volatility, analysis_time_s, video
    """
    df = pd.read_csv(path)

    int_cols = ["window_index"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    if "did_change" in df.columns:
        df["did_change"] = df["did_change"].astype(bool)

    float_cols = [
        "t_start", "t_end", "state_confidence",
        "valence", "arousal",
        "valence_volatility", "arousal_volatility",
        "analysis_time_s",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    return df


def load_all_sessions(session_csvs_dir) -> pd.DataFrame:
    """
    Load every CSV in session_csvs_dir, add a 'video' column from the filename
    stem (strips trailing _results), and return a single concatenated DataFrame.
    """
    session_csvs_dir = Path(session_csvs_dir)
    dfs: List[pd.DataFrame] = []

    for csv_path in sorted(session_csvs_dir.glob("*.csv")):
        df = load_session_csv(csv_path)
        stem = csv_path.stem
        if stem.endswith("_results"):
            stem = stem[: -len("_results")]
        df["video"] = stem
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ── Intent-sequence metrics ───────────────────────────────────────────────────

def compute_metrics(intent_sequence: List[str]) -> Dict:
    """
    Compute stability metrics for a sequence of effective intent strings.

    Returns:
        transition_rate — n_changes / n_windows
        flip_rate       — A→B→A reversals / n_windows
        entropy         — Shannon entropy (bits) over intent distribution
        intent_counts   — {intent: count}
    """
    n = len(intent_sequence)
    if n == 0:
        return {
            "transition_rate": 0.0,
            "flip_rate":       0.0,
            "entropy":         0.0,
            "intent_counts":   {},
        }

    # Transitions: any adjacent pair that differs
    n_changes = sum(
        1 for i in range(1, n)
        if intent_sequence[i] != intent_sequence[i - 1]
    )
    transition_rate = n_changes / n

    # Flips: A→B→A reversals (look-ahead by 2)
    n_flips = sum(
        1
        for i in range(1, n - 1)
        if (
            intent_sequence[i] != intent_sequence[i - 1]
            and intent_sequence[i + 1] == intent_sequence[i - 1]
        )
    )
    flip_rate = n_flips / n

    # Shannon entropy over intent distribution
    counts = Counter(intent_sequence)
    entropy = -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )

    return {
        "transition_rate": transition_rate,
        "flip_rate":       flip_rate,
        "entropy":         entropy,
        "intent_counts":   dict(counts),
    }


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_results(rows: List[Dict], path: str) -> None:
    """
    Write rows to a CSV at *path* using the built-in csv module.
    Overwrites any existing file.  Creates parent directories as needed.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
