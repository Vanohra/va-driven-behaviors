"""
Evaluation 2 — Confidence gating ablation.

Target:  online/online_session.py — ReactionHistory min_confidence threshold.
         The robot is NOT involved.

Purpose:
    Show that confidence gating suppresses low-quality windows and stabilises
    the effective intent sequence.

Method:
    For each session CSV in evaluation/session_csvs/:
        1. Replay proposed_intent + state_confidence + state_label through
           ReactionHistory with:
               A: min_confidence=0.0   (no gating)
               B: min_confidence=0.55  (production value from run_offline.py)
           Both conditions keep cooldown_s=0.5.
           Simulated time = t_start from each CSV row (never time.time()).
        2. Compute stability metrics + suppression_rate (fraction of windows
           where effective_intent fell back to NEUTRAL due to low confidence) +
           mean state_confidence at windows where did_change=True in the
           original session CSV.
        3. Save one row per (video, condition) to results.csv.

NOTE: pose_mode / va_pose_rate are NOT included.  pose_mode is determined by
SpotReactionMapper upstream of ReactionHistory and is unaffected by the
confidence threshold ablation.

Usage:
    python evaluation/eval_2_confidence_gating/run_eval.py
"""

import sys
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
PROD_COOLDOWN_S = 0.5    # run_offline.py --cooldown default

CONDITIONS = {
    "no_gating":   0.0,
    "with_gating": 0.55,
}


# ── Replay logic ──────────────────────────────────────────────────────────────

def replay_session(df, min_confidence: float, cooldown_s: float) -> dict:
    """
    Replay one session CSV through a fresh ReactionHistory.

    Uses t_start column as simulated current_time (never time.time()).

    Returns:
        effective_intents — list[str], one per window
        n_suppressed      — windows where reason == 'low_confidence'
    """
    history = ReactionHistory(min_confidence=min_confidence, cooldown_s=cooldown_s)
    effective_intents: list[str] = []
    n_suppressed = 0

    for _, row in df.iterrows():
        proposed = str(row["proposed_intent"])
        conf     = float(row["state_confidence"])
        label    = str(row["state_label"])
        t_start  = float(row["t_start"])   # simulated time — NOT time.time()

        effective, reason, _ = history.evaluate(proposed, conf, label, t_start)
        effective_intents.append(effective)
        if reason == "low_confidence":
            n_suppressed += 1

    return {
        "effective_intents": effective_intents,
        "n_suppressed":      n_suppressed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_csvs_dir: Path) -> None:
    csv_files = sorted(session_csvs_dir.glob("*.csv"))
    if not csv_files:
        print(f"[EVAL 2] No session CSVs found in {session_csvs_dir}")
        print("  Generate them first:")
        print("    for f in samples/*.mp4; do python run_offline.py \"$f\"; done")
        return

    rows = []
    for csv_path in csv_files:
        stem = csv_path.stem
        video_name = stem[: -len("_results")] if stem.endswith("_results") else stem

        df = load_session_csv(csv_path)
        n_windows = len(df)

        # mean state_confidence at windows where did_change=True in the original
        # session CSV (ground-truth from the full pipeline run)
        changed_df = df[df["did_change"] == True]
        mean_conf_at_transition = (
            float(changed_df["state_confidence"].mean())
            if len(changed_df) > 0 else 0.0
        )

        for cond_name, min_conf in CONDITIONS.items():
            result  = replay_session(df, min_confidence=min_conf,
                                     cooldown_s=PROD_COOLDOWN_S)
            metrics = compute_metrics(result["effective_intents"])
            suppression_rate = (
                result["n_suppressed"] / n_windows if n_windows > 0 else 0.0
            )
            rows.append({
                "video":                        video_name,
                "condition":                    cond_name,
                "transition_rate":              round(metrics["transition_rate"], 6),
                "flip_rate":                    round(metrics["flip_rate"], 6),
                "entropy":                      round(metrics["entropy"], 6),
                "suppression_rate":             round(suppression_rate, 6),
                "mean_confidence_at_transition": round(mean_conf_at_transition, 6),
                "n_windows":                    n_windows,
            })

        print(f"  [{video_name}] {n_windows} windows — done")

    save_results(rows, RESULTS_PATH)
    print(f"\n[EVAL 2] Saved {len(rows)} rows to {RESULTS_PATH}")
    print(f"  ({len(csv_files)} videos × {len(CONDITIONS)} conditions)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Eval 2: Confidence gating ablation"
    )
    p.add_argument(
        "--session-csvs", default=str(SESSION_CSVS), metavar="DIR",
        help=f"Directory containing session CSVs (default: {SESSION_CSVS})"
    )
    args = p.parse_args()
    run(Path(args.session_csvs))


if __name__ == "__main__":
    main()
