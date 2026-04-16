"""
Evaluation 4 — Adapter cooldown effect on robot command rate.

Target:  robot/bittle_adapter.py — INTENT_COOLDOWN_SEC = 2.5.
         This is the ONLY eval that involves robot behavior (in simulation —
         no physical robot required).

Purpose:
    Show that the hardware-level cooldown in BittleXAdapter reduces the rate
    of serial commands sent to the robot, preventing servo thrashing.  This
    is distinct from Eval 1, which measures pipeline intent stability.  Even a
    perfectly stable intent sequence dispatched faster than real-time (offline
    mode) would thrash servos without this cooldown.

Method:
    For each session CSV:
        1. Load the effective_intent sequence (already gated by ReactionHistory).
        2. Simulate BittleXAdapter dispatch through a pure function that mirrors
           _execute_intent logic, under:
               condition A: INTENT_COOLDOWN_SEC=0.0  (no adapter cooldown)
               condition B: INTENT_COOLDOWN_SEC=2.5  (production value)
           With two simulated timing modes:
               online_mode:  window interval = 3.0 s (realistic window spacing)
               offline_mode: window interval = 0.5 s (fast-replay simulation)
        3. Simulated time is tracked as a local float incremented by the
           window interval.  time.sleep is NEVER called.
        4. Record n_commands_sent, n_commands_dropped, command_rate, dedup_drop_rate,
           cooldown_drop_rate for each (video, condition, timing_mode) triple.

Usage:
    python evaluation/eval_4_adapter_cooldown/run_eval.py
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

from eval_utils import load_session_csv, save_results

# ── Production constants ──────────────────────────────────────────────────────
PROD_INTENT_COOLDOWN_SEC = 2.5   # bittle_adapter.py INTENT_COOLDOWN_SEC

CONDITIONS = {
    "no_cooldown":   0.0,
    "with_cooldown": PROD_INTENT_COOLDOWN_SEC,
}

TIMING_MODES = {
    "online_mode":  3.0,   # window interval = window_dur (run_offline.py default)
    "offline_mode": 0.5,   # fast-replay: 500 ms per window (sub-real-time processing)
}


# ── Pure simulation (no time.sleep, no hardware) ──────────────────────────────

def simulate_adapter_dispatch(
    intents: list[str],
    cooldown_sec: float,
    window_interval: float,
) -> dict:
    """
    Simulate BittleXAdapter._execute_intent dispatch for a sequence of intents.

    Mirrors the deduplication and rate-limiting logic inside _execute_intent:
        1. Dedup check  — if intent == current_pose, skip (no serial write needed)
        2. Cooldown check — if simulated_time - last_intent_time < cooldown_sec, skip
        3. Execute — update current_pose and last_intent_time

    Uses simulated_time (incremented by window_interval each step).
    NEVER calls time.sleep.

    Returns:
        n_commands_sent     — intents that passed both checks
        n_commands_dropped  — intents blocked (dedup + cooldown combined)
        command_rate        — n_commands_sent per minute of simulated time
        dedup_drop_rate     — dedup-dropped / n_windows
        cooldown_drop_rate  — cooldown-dropped / n_windows
    """
    current_pose: str | None = None
    last_intent_time: float = -float("inf")  # allows first intent to always pass cooldown
    simulated_time: float = 0.0

    n_sent = 0
    n_dedup = 0
    n_cooldown = 0

    for raw_intent in intents:
        intent = raw_intent.upper()

        # Dedup check (mirrors: if intent == self.current_pose: return)
        if intent == current_pose:
            n_dedup += 1
        # Cooldown check (mirrors: if now - self._last_intent_time < INTENT_COOLDOWN_SEC: return)
        elif (simulated_time - last_intent_time) < cooldown_sec:
            n_cooldown += 1
        else:
            # Execute: update state
            n_sent += 1
            current_pose     = intent
            last_intent_time = simulated_time

        simulated_time += window_interval

    n_windows = len(intents)
    total_sim_time = simulated_time  # = n_windows * window_interval

    command_rate = (
        n_sent / (total_sim_time / 60.0) if total_sim_time > 0 else 0.0
    )

    return {
        "n_commands_sent":    n_sent,
        "n_commands_dropped": n_dedup + n_cooldown,
        "command_rate":       round(command_rate, 4),
        "dedup_drop_rate":    round(n_dedup    / n_windows, 6) if n_windows > 0 else 0.0,
        "cooldown_drop_rate": round(n_cooldown / n_windows, 6) if n_windows > 0 else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_csvs_dir: Path) -> None:
    csv_files = sorted(session_csvs_dir.glob("*.csv"))
    if not csv_files:
        print(f"[EVAL 4] No session CSVs found in {session_csvs_dir}")
        print("  Generate them first:")
        print("    for f in samples/*.mp4; do python run_offline.py \"$f\"; done")
        return

    rows = []
    for csv_path in csv_files:
        stem = csv_path.stem
        video_name = stem[: -len("_results")] if stem.endswith("_results") else stem

        df = load_session_csv(csv_path)
        intents   = df["effective_intent"].tolist()
        n_windows = len(intents)

        for cond_name, cooldown_sec in CONDITIONS.items():
            for timing_name, window_interval in TIMING_MODES.items():
                sim = simulate_adapter_dispatch(
                    intents=intents,
                    cooldown_sec=cooldown_sec,
                    window_interval=window_interval,
                )
                rows.append({
                    "video":               video_name,
                    "condition":           cond_name,
                    "timing_mode":         timing_name,
                    "n_commands_sent":     sim["n_commands_sent"],
                    "n_commands_dropped":  sim["n_commands_dropped"],
                    "command_rate":        sim["command_rate"],
                    "dedup_drop_rate":     sim["dedup_drop_rate"],
                    "cooldown_drop_rate":  sim["cooldown_drop_rate"],
                    "n_windows":           n_windows,
                })

        print(f"  [{video_name}] {n_windows} windows — done")

    save_results(rows, RESULTS_PATH)
    print(f"\n[EVAL 4] Saved {len(rows)} rows to {RESULTS_PATH}")
    print(f"  ({len(csv_files)} videos × {len(CONDITIONS)} conditions "
          f"× {len(TIMING_MODES)} timing modes = {len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Eval 4: Adapter cooldown effect on robot command rate"
    )
    p.add_argument(
        "--session-csvs", default=str(SESSION_CSVS), metavar="DIR",
        help=f"Directory containing session CSVs (default: {SESSION_CSVS})"
    )
    args = p.parse_args()
    run(Path(args.session_csvs))


if __name__ == "__main__":
    main()
