"""
Script 02b: Phase × Engagement comparison plot

Posts a single figure showing per-hour engagement features stratified by
code-status phase. Reveals that labs, chart_decision, and consults drop
cleanly across full_code → dnr_active → cmo_active, while proc / vasoactive
/ vent_setting event counts are CONFOUNDED — they rise at CMO because
ramp-down activity (terminal extubation, line pulls, pressor weans) is
logged as procedures and titrations.

This informs the Script 03 decision of which features to include in the
"shift direction" training.
"""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vitrine import show

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

windows = pd.read_parquet(OUTPUT_DIR / f"02_windows_{DATE_TAG}.parquet")

clean_features = [
    ("lab_per_hr", "Labs / h"),
    ("chart_decision_per_hr", "Chart (decision) / h"),
    ("consult_new_per_hr", "New consults / h"),
]
ambiguous_features = [
    ("proc_per_hr", "Procedures / h"),
    ("vasoactive_per_hr", "Vasoactive events / h"),
    ("vent_setting_per_hr", "Vent setting events / h"),
]
phase_order = ["full_code", "dnr_active", "cmo_active"]
phase_colors = {"full_code": "#4C72B0", "dnr_active": "#DD8452", "cmo_active": "#C44E52"}

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=False)

def plot_means(ax, feat_col, title, phases):
    vals = [windows.loc[windows.phase == p, feat_col].mean() for p in phases]
    colors = [phase_colors[p] for p in phases]
    ax.bar(phases, vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax, (c, t) in zip(axes[0], clean_features):
    plot_means(ax, c, t, phase_order)
axes[0, 0].set_ylabel("mean per window", fontsize=10)

for ax, (c, t) in zip(axes[1], ambiguous_features):
    plot_means(ax, c, t, phase_order)
axes[1, 0].set_ylabel("mean per window", fontsize=10)

fig.suptitle(
    "Per-hour engagement × code-status phase\n"
    "Top row: clean 'less intensive' signal.  Bottom row: confounded by ramp-down activity (proc/titration events rise at CMO).",
    fontsize=11, y=1.02,
)
plt.tight_layout()
fig_path = OUTPUT_DIR / f"02b_phase_engagement_{DATE_TAG}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Wrote {fig_path}")

show(
    fig,
    title="Step 02 — Engagement × code-status phase",
    description=(
        "Key finding for Script 03: labs, decision-driven charting, and new "
        "consults all drop monotonically from full_code → dnr_active → cmo_active "
        "(top row — clean signal). But procedure, vasoactive, and vent-setting "
        "event counts RISE at cmo_active (bottom row — confounded by ramp-down "
        "activity like terminal extubation, line pulls, pressor weans). "
        "The shift-direction training in Script 03 should weight the clean "
        "features and either exclude or flip-sign the ambiguous ones."
    ),
    study=STUDY,
    source="scripts/02b_phase_comparison_plot_20260411.py",
)

plt.close(fig)
print("Done.")
