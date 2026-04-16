"""
Generate all paper-quality figures for the evaluation suite.
Saves 8 PNG files (300 DPI) to evaluation/figures/.

Run from project root:
    python evaluation/generate_figures.py
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
FIG_DIR  = EVAL_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

INTENT_COLORS = {
    "NEUTRAL":     "#B4B2A9",
    "CHECK_IN":    "#5DCAA5",
    "ENGAGE":      "#7F77DD",
    "DE_ESCALATE": "#D85A30",
    "CAUTION":     "#EF9F27",
}

# Global style
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


def save(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 1 — Figure A: Intent timeline (video49, most varied)
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval1_timeline():
    from online.online_session import ReactionHistory
    from eval_utils import load_all_sessions

    sessions = load_all_sessions(EVAL_DIR / "session_csvs")
    if sessions.empty:
        print("  [SKIP] No session CSVs for eval1 timeline")
        return

    # Pick video49 — most intent variety
    candidates = sessions.groupby("video")["window_index"].count()
    rep = "video49" if "video49" in candidates.index else candidates.idxmax()
    df = sessions[sessions["video"] == rep].sort_values("window_index").reset_index(drop=True)

    def replay(df, cooldown_s):
        h = ReactionHistory(min_confidence=0.55, cooldown_s=cooldown_s)
        intents = []
        for _, row in df.iterrows():
            e, _, _ = h.evaluate(str(row["proposed_intent"]),
                                  float(row["state_confidence"]),
                                  str(row["state_label"]),
                                  float(row["t_start"]))
            intents.append(e)
        return intents

    no_cd   = replay(df, 0.0)
    with_cd = replay(df, 0.5)
    n = len(no_cd)

    fig, axes = plt.subplots(2, 1, figsize=(7, 2.4), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ax, intents, label in [
        (axes[0], no_cd,   r"No cooldown ($\delta_t = 0.0$ s)"),
        (axes[1], with_cd, r"Production cooldown ($\delta_t = 0.5$ s)"),
    ]:
        for i, intent in enumerate(intents):
            ax.barh(0, 1, left=i, color=INTENT_COLORS.get(intent, "#888"),
                    height=0.55, align="center", linewidth=0)
        ax.set_yticks([])
        ax.set_xlim(0, n)
        ax.set_ylabel(label, fontsize=8.5, rotation=0, labelpad=4,
                      ha="left", va="center")
        ax.yaxis.set_label_coords(-0.01, 0.5)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Window index", fontsize=9)
    fig.suptitle(
        f"Intent timeline — {rep}   (conditions identical at 3 s window spacing)",
        fontsize=10, y=1.01
    )

    patches = [mpatches.Patch(color=v, label=k) for k, v in INTENT_COLORS.items()
               if k in no_cd]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               bbox_to_anchor=(0.5, -0.18), fontsize=8.5, frameon=False)
    save(fig, "eval1_timeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 1 — Figure B: Transition + flip rate per condition
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval1_metrics():
    df = pd.read_csv(EVAL_DIR / "eval_1_cooldown" / "results.csv")
    cond_order  = ["no_cooldown", "with_cooldown"]
    cond_labels = ["No cooldown\n($\\delta_t=0.0$ s)", "Prod. cooldown\n($\\delta_t=0.5$ s)"]
    metrics     = ["transition_rate", "flip_rate"]
    mlabels     = ["Transition rate\n(changes / windows)", "Flip rate\n(A→B→A / windows)"]
    colors      = ["#90C8E8", "#E8A090"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    for ax, metric, mlabel, color in zip(axes, metrics, mlabels, colors):
        means = df.groupby("condition")[metric].mean().reindex(cond_order)
        stds  = df.groupby("condition")[metric].std().reindex(cond_order)
        bars  = ax.bar(range(len(cond_order)), means, yerr=stds, capsize=5,
                       color=[color, color], edgecolor="k", linewidth=0.6, width=0.5)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_labels, fontsize=8.5)
        ax.set_ylabel(mlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, (bar, val, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std * 0.05 + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Eval 1: Cooldown ablation — stability metrics across 11 videos",
                 fontsize=10)
    plt.tight_layout()
    save(fig, "eval1_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 2 — Figure A: Confidence scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval2_scatter():
    from eval_utils import load_all_sessions
    sessions = load_all_sessions(EVAL_DIR / "session_csvs")
    if sessions.empty:
        print("  [SKIP] No session CSVs for eval2 scatter")
        return

    THRESHOLD = 0.55
    sessions["passed"] = sessions["state_confidence"] >= THRESHOLD

    intent_order = ["NEUTRAL", "CHECK_IN", "ENGAGE", "DE_ESCALATE", "CAUTION"]
    present = [i for i in intent_order if i in sessions["proposed_intent"].unique()]
    x_pos = {intent: i for i, intent in enumerate(present)}

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    rng = np.random.default_rng(42)
    for _, row in sessions.iterrows():
        xi = x_pos.get(str(row["proposed_intent"]), -1)
        if xi < 0:
            continue
        jit = rng.uniform(-0.18, 0.18)
        color = "#2CA02C" if row["passed"] else "#D62728"
        ax.scatter(xi + jit, float(row["state_confidence"]),
                   color=color, alpha=0.35, s=14, linewidths=0)

    ax.axhline(THRESHOLD, color="navy", linestyle="--", linewidth=1.4,
               label=f"Confidence threshold = {THRESHOLD}")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=15, ha="right")
    ax.set_ylabel("state_confidence")
    ax.set_xlabel("Proposed intent")
    ax.set_title("Eval 2: State confidence vs proposed intent — all windows, 11 videos")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    passed_patch   = mpatches.Patch(color="#2CA02C", alpha=0.7, label="Passed (conf ≥ 0.55)")
    suppress_patch = mpatches.Patch(color="#D62728", alpha=0.7, label="Suppressed (conf < 0.55)")
    ax.legend(handles=[passed_patch, suppress_patch,
                        plt.Line2D([0],[0], color="navy", linestyle="--", label=f"Threshold = {THRESHOLD}")],
              fontsize=8.5, frameon=False)
    plt.tight_layout()
    save(fig, "eval2_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 2 — Figure B: Suppression rate per video
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval2_suppression():
    df = pd.read_csv(EVAL_DIR / "eval_2_confidence_gating" / "results.csv")
    gated = df[df["condition"] == "with_gating"].sort_values("suppression_rate",
                                                              ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    bars = ax.bar(range(len(gated)), gated["suppression_rate"],
                  color="#D85A30", edgecolor="k", linewidth=0.5, width=0.7)
    mean_sr = gated["suppression_rate"].mean()
    ax.axhline(mean_sr, color="navy", linestyle="--", linewidth=1.3,
               label=f"Mean = {mean_sr:.2f}")
    ax.set_xticks(range(len(gated)))
    ax.set_xticklabels(gated["video"].tolist(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Suppression rate  (windows gated to NEUTRAL / total windows)")
    ax.set_ylim(0, 1.08)
    ax.set_title(r"Eval 2: Confidence suppression rate per video  ($\theta_c = 0.55$)")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, gated["suppression_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    save(fig, "eval2_suppression.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 3 — Figure A: Raw vs preprocessed time series overlay
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval3_timeseries():
    npz_path = EVAL_DIR / "eval_3_va_preprocessing" / "eval_3_sample_series.npz"
    if not npz_path.exists():
        print("  [SKIP] eval_3_sample_series.npz not found")
        return
    data = np.load(npz_path, allow_pickle=True)
    vid  = str(data["video_name"])
    vraw = data["valence_raw"]
    araw = data["arousal_raw"]
    vproc = data["valence_processed"]
    aproc = data["arousal_processed"]
    frames = np.arange(len(vraw))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    fig.subplots_adjust(hspace=0.12)

    for ax, raw, proc, ylabel, proc_col in [
        (axes[0], vraw, vproc, "Valence", "#7F77DD"),
        (axes[1], araw, aproc, "Arousal", "#5DCAA5"),
    ]:
        ax.plot(frames, raw,  color="#C8C8C8", linewidth=0.7, alpha=0.95,
                label="Raw (pre-preprocessing)", zorder=1)
        ax.plot(frames, proc, color=proc_col,   linewidth=1.8,
                label="Preprocessed (Winsor + Hampel + rolling-median)", zorder=2)
        ax.axhline(0, color="k", linewidth=0.4, linestyle=":")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(
        f"Eval 3: Raw vs preprocessed VA signal — {vid}  (300 frames, 3 s window)",
        fontsize=10
    )
    save(fig, "eval3_timeseries.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 3 — Figure B: MAD / IQR / frame_delta_variance grouped bars
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval3_metrics():
    df = pd.read_csv(EVAL_DIR / "eval_3_va_preprocessing" / "results.csv")
    by_cond = (df.groupby(["video", "condition"])[["mad", "iqr", "frame_delta_variance"]]
               .mean().reset_index())

    cond_order  = ["raw", "processed"]
    cond_colors = ["#C0C0C0", "#7F77DD"]
    metrics     = ["mad", "iqr", "frame_delta_variance"]
    mlabels     = ["MAD", "IQR", r"Frame $\Delta$ variance"]

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.4))
    for ax, metric, mlabel, color_pair in zip(axes, metrics, mlabels,
                                               [cond_colors]*3):
        means = by_cond.groupby("condition")[metric].mean().reindex(cond_order)
        stds  = by_cond.groupby("condition")[metric].std().reindex(cond_order)
        bars  = ax.bar(range(len(cond_order)), means, yerr=stds, capsize=5,
                       color=color_pair, edgecolor="k", linewidth=0.6, width=0.52)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_order)
        ax.set_title(mlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Reduction %
        if means["raw"] > 0:
            pct = (means["raw"] - means["processed"]) / means["raw"] * 100
            direction = "↓" if pct > 0 else "↑"
            ax.text(0.5, 0.92, f"{direction}{abs(pct):.1f}%",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    color="#333", style="italic")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.5f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Eval 3: Signal quality — raw vs preprocessed  (mean ± std, 5 videos × 2 channels)",
                 fontsize=10)
    plt.tight_layout()
    save(fig, "eval3_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 4 — Figure A: Command rate grouped bar chart (KEY FIGURE)
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval4_commandrate():
    df = pd.read_csv(EVAL_DIR / "eval_4_adapter_cooldown" / "results.csv")
    cond_order   = ["no_cooldown", "with_cooldown"]
    timing_order = ["online_mode", "offline_mode"]
    timing_colors = {"online_mode": "#90C8E8", "offline_mode": "#E8A090"}
    timing_labels = {"online_mode": "Online mode\n(3.0 s / window)",
                     "offline_mode": "Offline mode\n(0.5 s / window)"}

    pivot_mean = (df.groupby(["condition", "timing_mode"])["command_rate"]
                  .mean().unstack("timing_mode")
                  .reindex(index=cond_order, columns=timing_order))
    pivot_std  = (df.groupby(["condition", "timing_mode"])["command_rate"]
                  .std().unstack("timing_mode")
                  .reindex(index=cond_order, columns=timing_order))

    x     = np.arange(len(cond_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for i, (timing, color) in enumerate(timing_colors.items()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, pivot_mean[timing], width,
                      yerr=pivot_std[timing], capsize=5,
                      color=color, edgecolor="k", linewidth=0.6,
                      label=timing_labels[timing])
        for bar, val in zip(bars, pivot_mean[timing]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([r"No adapter cooldown" + "\n" + r"($\tau=0.0$ s)",
                         r"With adapter cooldown" + "\n" + r"($\tau=2.5$ s)"],
                        fontsize=9.5)
    ax.set_ylabel("Mean command rate  (commands / minute)", fontsize=9.5)
    ax.set_title("Eval 4: Robot command rate — adapter cooldown ablation\n"
                 r"(mean $\pm$ std across 11 videos)", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save(fig, "eval4_commandrate.png")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL 4 — Figure B: Stacked proportions
# ─────────────────────────────────────────────────────────────────────────────

def fig_eval4_stacked():
    df = pd.read_csv(EVAL_DIR / "eval_4_adapter_cooldown" / "results.csv")
    df["sent_rate"]    = df["n_commands_sent"]    / df["n_windows"]
    df["dedup_rate"]   = df["dedup_drop_rate"]
    df["cool_rate"]    = df["cooldown_drop_rate"]

    cond_order   = ["no_cooldown", "with_cooldown"]
    timing_order = ["online_mode", "offline_mode"]
    timing_labels = {"online_mode": "Online\n(3.0 s/win)",
                     "offline_mode": "Offline\n(0.5 s/win)"}

    stack_cols   = ["sent_rate",   "dedup_rate",    "cool_rate"]
    stack_colors = ["#5DCAA5",     "#B4B2A9",        "#EF9F27"]
    stack_labels = ["Fired",       "Dropped: dedup", "Dropped: cooldown"]

    group = df.groupby(["condition", "timing_mode"])[stack_cols].mean()

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, timing in zip(axes, timing_order):
        sub = group.xs(timing, level="timing_mode").reindex(cond_order)
        bottom = np.zeros(len(cond_order))
        for col, color, label in zip(stack_cols, stack_colors, stack_labels):
            vals = sub[col].values
            ax.bar(range(len(cond_order)), vals, bottom=bottom,
                   color=color, label=label, edgecolor="k",
                   linewidth=0.5, width=0.52)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.04:
                    ax.text(xi, b + v / 2, f"{v:.2f}",
                            ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold")
            bottom += vals
        ax.set_title(timing_labels[timing], fontsize=9.5)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(["No\ncooldown", "With\ncooldown\n(2.5 s)"],
                            fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction of windows", fontsize=9.5)
    fig.suptitle("Eval 4: Command fate — fired vs dropped per condition",
                 fontsize=10, y=1.01)

    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(stack_colors, stack_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.1), fontsize=9, frameon=False)
    save(fig, "eval4_stacked.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR}\n")
    fig_eval1_timeline()
    fig_eval1_metrics()
    fig_eval2_scatter()
    fig_eval2_suppression()
    fig_eval3_timeseries()
    fig_eval3_metrics()
    fig_eval4_commandrate()
    fig_eval4_stacked()
    print(f"\nDone. {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")
