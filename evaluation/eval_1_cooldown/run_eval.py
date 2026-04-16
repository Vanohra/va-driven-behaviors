"""
Evaluation 1 — ReactionHistory cooldown ablation.

Target:  online/online_session.py — ReactionHistory cooldown_s parameter.
         NOT the hardware adapter cooldown in bittle_adapter.py (that is Eval 4).
         The robot is NOT involved.

Purpose:
    Show that the pipeline-level cooldown in ReactionHistory reduces intent
    jitter (rapid transitions and reversals) without suppressing genuine changes.

Method:
    For each session CSV in evaluation/session_csvs/:
        1. Load the CSV.
        2. Replay proposed_intent + state_confidence through a fresh
           ReactionHistory under two conditions:
               A: cooldown_s=0.0   (no cooldown)
               B: cooldown_s=0.5   (production value from run_offline.py)
           Both conditions use min_confidence=0.55.
        3. Use t_start from the CSV row as simulated current_time (never
           time.time()).  This evaluates the 0.5 s cooldown against the
           actual 3.0 s window spacing — the realistic production scenario.
        4. Compute stability metrics (compute_metrics from eval_utils) on
           each resulting effective_intent sequence.
        5. Save one row per (video, condition) to results.csv.

Usage:
    python evaluation/eval_1_cooldown/run_eval.py
    python evaluation/eval_1_cooldown/run_eval.py --session-csvs path/to/csvs
"""

import sys
import json
import argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
EVAL_DIR     = Path(__file__).resolve().parent.parent   # evaluation/
ROOT         = EVAL_DIR.parent                           # project root
SESSION_CSVS = EVAL_DIR / "session_csvs"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from online.online_session import ReactionHistory
from eval_utils import load_session_csv, compute_metrics, save_results

# ── Production constants ──────────────────────────────────────────────────────
PROD_MIN_CONFIDENCE = 0.55   # run_offline.py --min-confidence default
PROD_COOLDOWN_S     = 0.5    # run_offline.py --cooldown default

CONDITIONS = {
    "no_cooldown":   0.0,
    "with_cooldown": PROD_COOLDOWN_S,
}


# ── Replay logic ──────────────────────────────────────────────────────────────

def replay_session(df, cooldown_s: float, min_confidence: float) -> dict:
    """
    Replay one session CSV through a fresh ReactionHistory.

    Uses t_start column as simulated current_time so the cooldown is
    evaluated against realistic window spacing (never time.time()).

    Returns:
        effective_intents        — list[str], one per window
        n_suppressed_by_cooldown — int, windows blocked by the cooldown gate
    """
    history = ReactionHistory(min_confidence=min_confidence, cooldown_s=cooldown_s)
    effective_intents: list[str] = []
    n_suppressed_by_cooldown = 0

    for _, row in df.iterrows():
        proposed = str(row["proposed_intent"])
        conf     = float(row["state_confidence"])
        label    = str(row["state_label"])
        t_start  = float(row["t_start"])   # simulated time — NOT time.time()

        effective, reason, _ = history.evaluate(proposed, conf, label, t_start)
        effective_intents.append(effective)
        if reason == "cooldown":
            n_suppressed_by_cooldown += 1

    return {
        "effective_intents":        effective_intents,
        "n_suppressed_by_cooldown": n_suppressed_by_cooldown,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_csvs_dir: Path) -> None:
    csv_files = sorted(session_csvs_dir.glob("*.csv"))
    if not csv_files:
        print(f"[EVAL 1] No session CSVs found in {session_csvs_dir}")
        print("  Generate them first:")
        print("    for f in samples/*.mp4; do python run_offline.py \"$f\"; done")
        return

    rows = []
    for csv_path in csv_files:
        stem = csv_path.stem
        video_name = stem[: -len("_results")] if stem.endswith("_results") else stem

        df = load_session_csv(csv_path)
        n_windows = len(df)

        for cond_name, cooldown_s in CONDITIONS.items():
            result  = replay_session(df, cooldown_s=cooldown_s,
                                     min_confidence=PROD_MIN_CONFIDENCE)
            metrics = compute_metrics(result["effective_intents"])
            rows.append({
                "video":                    video_name,
                "condition":                cond_name,
                "transition_rate":          round(metrics["transition_rate"], 6),
                "flip_rate":                round(metrics["flip_rate"], 6),
                "entropy":                  round(metrics["entropy"], 6),
                "n_windows":                n_windows,
                "n_suppressed_by_cooldown": result["n_suppressed_by_cooldown"],
                "intent_counts":            json.dumps(metrics["intent_counts"]),
            })

        print(f"  [{video_name}] {n_windows} windows — done")

    save_results(rows, RESULTS_PATH)
    print(f"\n[EVAL 1] Saved {len(rows)} rows to {RESULTS_PATH}")
    print(f"  ({len(csv_files)} videos × {len(CONDITIONS)} conditions)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Eval 1: ReactionHistory cooldown ablation"
    )
    p.add_argument(
        "--session-csvs", default=str(SESSION_CSVS), metavar="DIR",
        help=f"Directory containing session CSVs (default: {SESSION_CSVS})"
    )
    args = p.parse_args()
    run(Path(args.session_csvs))


if __name__ == "__main__":
    main()
