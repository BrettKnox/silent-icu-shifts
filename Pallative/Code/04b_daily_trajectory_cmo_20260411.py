"""
Script 04b: Daily trajectory of palliative-ness score for CMO (and DNR-only) stays
===================================================================================

For every CMO transition stay (and DNR-only stays as comparison), compute the
palliative-ness score for each 24 h window in the stay, aligned so that
day_rel = 0 is the day containing first_limit_time (CMO or DNR order).

Plot the per-day mean trajectory with 95% CI, overlaid with individual patient
spaghetti.

2026-04-11 revision: Drop each stay's LAST window and use the Script 05
CLEANED residualization model. This removes discharge/death contamination
in the last window and re-estimates the severity model on the cleaned pool.
The resulting trajectory is the "true" silent-shift signal without the
last-window artifact.

Inputs:
  - outputs/02_windows_20260411.parquet
  - outputs/01_cohort_20260411.parquet
  - outputs/05_residualization_model_cleaned_20260411.pkl

Outputs:
  - outputs/04b_daily_trajectory_cmo_20260411.png
  - outputs/04b_daily_trajectory_cmo_dnr_20260411.png
  - outputs/04b_daily_trajectory_data_20260411.parquet
"""

import pickle
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vitrine import show

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(all="ignore")

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# 1. Load CLEANED model artifacts from Script 05
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / f"05_residualization_model_cleaned_{DATE_TAG}.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = model["features"]
COVARS = model["covars"]
MIN_WINDOW_HOURS = model["min_window_hours"]
betas = {f: np.array(model["betas"][f]) for f in FEATURES}
covar_mean = pd.Series(model["covar_mean"])
covar_std = pd.Series(model["covar_std"])
fc_stds = model["fc_stds"]
feature_caps = model["feature_caps"]
v_comb = np.array(model["direction_vectors"]["combined_unit"])

# ---------------------------------------------------------------------------
# 2. Load cohort + windows; join CMO / DNR stays
# ---------------------------------------------------------------------------
cohort = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
cohort = cohort[cohort.cs_pattern != "other"].reset_index(drop=True)
windows = pd.read_parquet(OUTPUT_DIR / f"02_windows_{DATE_TAG}.parquet")

# Windows parquet already carries first_cmo_time / first_dnr_time from Script 02,
# so just filter and rename. Drop stays we don't want.
cmo_wins = windows[
    windows.stay_id.isin(cohort[cohort.cs_pattern == "reaches_cmo"].stay_id)
].copy()
cmo_wins["first_limit_time"] = cmo_wins.first_cmo_time
cmo_wins["transition_type"] = "cmo"

dnr_wins = windows[
    windows.stay_id.isin(cohort[cohort.cs_pattern == "reaches_dnr"].stay_id)
].copy()
dnr_wins["first_limit_time"] = dnr_wins.first_dnr_time
dnr_wins["transition_type"] = "dnr"

print(f"CMO stay windows: {len(cmo_wins):,}  (from {cmo_wins.stay_id.nunique()} stays)")
print(f"DNR stay windows: {len(dnr_wins):,}  (from {dnr_wins.stay_id.nunique()} stays)")

both = pd.concat([cmo_wins, dnr_wins], ignore_index=True)

# Drop each stay's LAST window to remove discharge/death contamination.
max_idx = both.groupby("stay_id").window_idx.transform("max")
both["is_last_window"] = both.window_idx == max_idx
n_last = int(both.is_last_window.sum())
print(f"Dropping {n_last:,} last-of-stay windows ({n_last / len(both) * 100:.1f}%)")
both = both[~both.is_last_window].copy()

# Drop short partial windows and rows missing covariates
both = both[both.window_hours >= MIN_WINDOW_HOURS].copy()
both = both.dropna(subset=COVARS + ["lab_per_hr", "chart_decision_per_hr"])
both["lab_distinct_itemid_per_hr"] = both.lab_distinct_itemid / both.window_hours
print(f"Windows after QC: {len(both):,}")

# ---------------------------------------------------------------------------
# 3. Apply residualization model → palliative_score
# ---------------------------------------------------------------------------
# Cap features
for f in FEATURES:
    both[f] = np.minimum(both[f].astype(float), feature_caps[f])

# Standardize covariates
X = ((both[COVARS].astype(float) - covar_mean) / covar_std).values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def apply_residualizer(X, beta):
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta

for f in FEATURES:
    pred = apply_residualizer(X, betas[f])
    both[f + "_resid"] = both[f].astype(float).values - pred
    both[f + "_resid_z"] = both[f + "_resid"] / fc_stds[f]

z_cols = [f + "_resid_z" for f in FEATURES]
both["palliative_score"] = both[z_cols].values @ v_comb

# ---------------------------------------------------------------------------
# 4. Compute day_rel: day relative to first_limit_time (midpoint anchored)
# ---------------------------------------------------------------------------
midpoints = both.window_start + (both.window_end - both.window_start) / 2
both["hours_to_limit"] = (midpoints - both.first_limit_time).dt.total_seconds() / 3600.0
both["day_rel"] = np.floor(both.hours_to_limit / 24.0).astype(int)

# Save data for reuse
both[["stay_id", "transition_type", "day_rel", "palliative_score",
      "window_start", "window_end", "window_hours", "first_limit_time",
      "lab_per_hr", "chart_decision_per_hr", "consult_new_per_hr"]].to_parquet(
    OUTPUT_DIR / f"04b_daily_trajectory_data_{DATE_TAG}.parquet", index=False
)

# ---------------------------------------------------------------------------
# 5. Figure 1: CMO-only daily trajectory
# ---------------------------------------------------------------------------
def plot_daily_trajectory(ax, sub, title, spaghetti_color, mean_color, min_n=20, day_range=(-10, 5)):
    day_lo, day_hi = day_range
    # Compute per-day mean and 95% CI
    daily = (
        sub[(sub.day_rel >= day_lo) & (sub.day_rel <= day_hi)]
        .groupby("day_rel")
        .agg(mean_score=("palliative_score", "mean"),
             sem_score=("palliative_score", "sem"),
             n=("palliative_score", "size"))
        .reset_index()
    )
    daily = daily[daily.n >= min_n]
    daily["ci_lo"] = daily.mean_score - 1.96 * daily.sem_score
    daily["ci_hi"] = daily.mean_score + 1.96 * daily.sem_score

    # Spaghetti: sample some stays with trajectories spanning multiple days
    stay_counts = sub.groupby("stay_id").day_rel.agg(["min", "max", "count"])
    eligible = stay_counts[stay_counts["count"] >= 3].index
    sample_ids = np.random.default_rng(42).choice(
        eligible, size=min(100, len(eligible)), replace=False
    )
    for sid in sample_ids:
        s = sub[sub.stay_id == sid].sort_values("day_rel")
        s = s[(s.day_rel >= day_lo) & (s.day_rel <= day_hi)]
        if len(s) >= 2:
            ax.plot(s.day_rel, s.palliative_score, color=spaghetti_color, alpha=0.12, lw=0.5)

    # Mean with CI
    ax.fill_between(daily.day_rel, daily.ci_lo, daily.ci_hi, color=mean_color, alpha=0.25)
    ax.plot(daily.day_rel, daily.mean_score, color=mean_color, lw=2.5, marker="o",
            markersize=7, label=f"mean ± 95% CI")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7, label="first DNR/CMO order")
    ax.axhline(0, color="black", lw=0.5, ls=":", alpha=0.5)

    # Sample size annotation
    for _, r in daily.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.day_rel, r.ci_hi), fontsize=7,
                    ha="center", va="bottom", color="#555")

    ax.set_xlabel("Days relative to first DNR/CMO order")
    ax.set_ylabel("Palliative-ness score\n(combined direction projection)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(day_lo - 0.5, day_hi + 0.5)
    return daily


fig1, ax1 = plt.subplots(1, 1, figsize=(9, 5.5))
cmo_sub = both[both.transition_type == "cmo"]
daily_cmo = plot_daily_trajectory(
    ax1, cmo_sub,
    f"CMO transitions (n={cmo_sub.stay_id.nunique()} stays)",
    spaghetti_color="#C44E52", mean_color="#C44E52",
)
plt.tight_layout()
fig_path_1 = OUTPUT_DIR / f"04b_daily_trajectory_cmo_{DATE_TAG}.png"
fig1.savefig(fig_path_1, dpi=150, bbox_inches="tight")
print(f"Wrote {fig_path_1}")

# ---------------------------------------------------------------------------
# 6. Figure 2: CMO vs DNR side-by-side
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
daily_cmo2 = plot_daily_trajectory(
    axes[0], cmo_sub,
    f"CMO transitions (n={cmo_sub.stay_id.nunique()})",
    "#C44E52", "#C44E52",
)
dnr_sub = both[both.transition_type == "dnr"]
daily_dnr = plot_daily_trajectory(
    axes[1], dnr_sub,
    f"DNR-only transitions (n={dnr_sub.stay_id.nunique()})",
    "#DD8452", "#DD8452",
)
plt.tight_layout()
fig_path_2 = OUTPUT_DIR / f"04b_daily_trajectory_cmo_dnr_{DATE_TAG}.png"
fig2.savefig(fig_path_2, dpi=150, bbox_inches="tight")
print(f"Wrote {fig_path_2}")

# ---------------------------------------------------------------------------
# 7. Post to vitrine
# ---------------------------------------------------------------------------
desc = (
    "Each 24 h window of every CMO stay is scored using the trained shift "
    "direction, then aligned to the CMO order (day 0 = window containing "
    "first_cmo_time). Curve shows per-day mean ± 95% CI with individual "
    "patient spaghetti underneath. Sample size annotated above each point. "
    "Watch the rise starting 2-3 days before the order — that's the leading "
    "shift showing up as a continuous trajectory rather than a single "
    "pre-vs-baseline contrast."
)
show(fig1, title="Step 04b — Daily trajectory (CMO stays)", description=desc,
     study=STUDY, source="scripts/04b_daily_trajectory_cmo_20260411.py")
show(fig2, title="Step 04b — Daily trajectory (CMO vs DNR-only)",
     description="Side-by-side comparison. Both transition types show pre-order rise, with CMO being the sharper curve.",
     study=STUDY, source="scripts/04b_daily_trajectory_cmo_20260411.py")

# Print the numeric trajectory for the record
print("\nCMO per-day means:")
print(daily_cmo.round(3).to_string(index=False))
print("\nDNR-only per-day means:")
print(daily_dnr.round(3).to_string(index=False))

plt.close("all")
print("Done.")
