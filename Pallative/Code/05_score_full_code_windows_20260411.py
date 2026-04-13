"""
Script 05: Score All Full Code Windows in the Cohort
======================================================

Apply a residualization model to every Full Code window and compute the
palliative-ness score, dropping each stay's LAST window to avoid
discharge/death/transfer contamination (confirmed with researcher 2026-04-11:
last-window scores are systematically elevated because end-of-stay triggers
a burst of documentation and order activity).

Because the last window is dropped, we RE-FIT the severity residualization
model on the cleaned Full Code pool rather than reusing Script 04's model
(which was trained on the uncleaned pool). The direction vector from
Script 04 is reused unchanged because it was learned from within-patient
paired deltas that are independent of the last-window contamination.

Covers three groups of Full Code windows:
  1. never_transition: stays with no DNR/CMO ever charted
  2. pre_dnr:          Full Code windows of stays that later transitioned to DNR-only
  3. pre_cmo:          Full Code windows of stays that later transitioned to CMO

Inputs:
  - outputs/02_windows_20260411.parquet
  - outputs/01_cohort_20260411.parquet
  - outputs/04_residualization_model_20260411.pkl  (for FEATURES, COVARS,
    feature_caps, direction vector; betas/means/stds are re-fit here)

Outputs:
  - outputs/05_full_code_scored_20260411.parquet  (per-window scored)
  - outputs/05_stay_level_scores_20260411.parquet
  - outputs/05_residualization_model_cleaned_20260411.pkl  (refit artifacts)
  - outputs/05_score_histogram_20260411.png
  - outputs/05_score_by_group_20260411.png
"""

import pickle
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vitrine import section, show

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(all="ignore")

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

section("Step 05: Score All Full Code Windows", study=STUDY)

# ---------------------------------------------------------------------------
# 1. Load artifacts (we REUSE features/covars/caps/direction, REFIT the rest)
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / f"04_residualization_model_{DATE_TAG}.pkl", "rb") as f:
    model_old = pickle.load(f)

FEATURES = model_old["features"]
COVARS = model_old["covars"]
MIN_WINDOW_HOURS = model_old["min_window_hours"]
feature_caps = model_old["feature_caps"]
v_comb = np.array(model_old["direction_vectors"]["combined_unit"])
v_cmo = np.array(model_old["direction_vectors"]["cmo_unit"])
v_dnr = np.array(model_old["direction_vectors"]["dnr_unit"])

cohort = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
cohort = cohort[cohort.cs_pattern != "other"].reset_index(drop=True)
windows = pd.read_parquet(OUTPUT_DIR / f"02_windows_{DATE_TAG}.parquet")

# ---------------------------------------------------------------------------
# 2. Filter to Full Code windows, drop each stay's LAST window,
#    and apply QC (window_hours>=2, covariate dropna)
# ---------------------------------------------------------------------------
fc_all = windows[windows.phase == "full_code"].copy()
max_idx = fc_all.groupby("stay_id").window_idx.transform("max")
fc_all["is_last_window"] = fc_all.window_idx == max_idx

n_total = len(fc_all)
n_last = int(fc_all.is_last_window.sum())
print(f"Full Code windows before drop: {n_total:,}")
print(f"  Dropping last-of-stay windows: {n_last:,}")
fc_windows = fc_all[~fc_all.is_last_window].copy()
fc_windows = fc_windows[fc_windows.window_hours >= MIN_WINDOW_HOURS]
fc_windows = fc_windows.dropna(subset=COVARS + ["lab_per_hr", "chart_decision_per_hr"])
print(f"  After ≥{MIN_WINDOW_HOURS}h + covariate dropna: {len(fc_windows):,}")
print(f"Unique stays: {fc_windows.stay_id.nunique():,}")

# Add lab_distinct_itemid_per_hr (not pre-computed in 02 parquet)
fc_windows["lab_distinct_itemid_per_hr"] = fc_windows.lab_distinct_itemid / fc_windows.window_hours

# Apply feature caps (same caps as Script 04 for consistency with direction vector)
for f in FEATURES:
    fc_windows[f] = np.minimum(fc_windows[f].astype(float), feature_caps[f])

# ---------------------------------------------------------------------------
# 2b. REFIT severity residualization model on the CLEANED Full Code pool
# ---------------------------------------------------------------------------
print("\nRe-fitting residualization model on CLEANED Full Code pool (last-window dropped)...")

covar_mean = fc_windows[COVARS].astype(float).mean()
covar_std = fc_windows[COVARS].astype(float).std().replace(0, 1.0)

def standardize(df, cols, mean, std):
    return ((df[cols].astype(float) - mean) / std).values

X_fit = standardize(fc_windows, COVARS, covar_mean, covar_std)

def fit_residualizer(X, y):
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta

def apply_residualizer(X, beta):
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta

betas = {}
fc_means = {}
fc_stds = {}
for f in FEATURES:
    y = fc_windows[f].astype(float).values
    betas[f] = fit_residualizer(X_fit, y)
    fc_means[f] = float(y.mean())
    fc_stds[f] = float(y.std()) or 1.0

# Save cleaned artifacts
cleaned_model = {
    "features": FEATURES,
    "covars": COVARS,
    "min_window_hours": MIN_WINDOW_HOURS,
    "feature_caps": feature_caps,
    "betas": {f: betas[f].tolist() for f in FEATURES},
    "covar_mean": covar_mean.to_dict(),
    "covar_std": covar_std.to_dict(),
    "fc_means": fc_means,
    "fc_stds": fc_stds,
    "direction_vectors": {
        "cmo_unit": v_cmo.tolist(),
        "dnr_unit": v_dnr.tolist(),
        "combined_unit": v_comb.tolist(),
    },
    "metadata": {
        "study": STUDY,
        "date": DATE_TAG,
        "fc_rows_used_for_fit": int(len(fc_windows)),
        "note": "Refit on cleaned pool (last-window dropped). Direction vector reused from Script 04.",
    },
}
with open(OUTPUT_DIR / f"05_residualization_model_cleaned_{DATE_TAG}.pkl", "wb") as f_out:
    pickle.dump(cleaned_model, f_out)
print(f"Saved cleaned residualization model.")

# ---------------------------------------------------------------------------
# 3. Residualize and project → palliative_score (using refit model)
# ---------------------------------------------------------------------------
X = standardize(fc_windows, COVARS, covar_mean, covar_std)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

for f in FEATURES:
    pred = apply_residualizer(X, betas[f])
    fc_windows[f + "_resid"] = fc_windows[f].astype(float).values - pred
    fc_windows[f + "_resid_z"] = fc_windows[f + "_resid"] / fc_stds[f]

z_cols = [f + "_resid_z" for f in FEATURES]
fc_windows["palliative_score"] = fc_windows[z_cols].values @ v_comb

# ---------------------------------------------------------------------------
# 4. Label windows by downstream group (never_transition / pre_dnr / pre_cmo)
# ---------------------------------------------------------------------------
pattern_map = cohort.set_index("stay_id")["cs_pattern"]
fc_windows["cs_pattern"] = fc_windows.stay_id.map(pattern_map)
fc_windows["group"] = fc_windows.cs_pattern.map({
    "no_cs_documented": "never_transition",
    "only_fullcode": "never_transition",
    "reaches_dnr": "pre_dnr",
    "reaches_cmo": "pre_cmo",
})

# Save scored parquet
out_cols = [
    "stay_id", "subject_id", "hadm_id", "window_idx", "window_start", "window_end",
    "window_hours", "group", "cs_pattern", "palliative_score",
    "sofa_max", "sofa_mean", "vent_any_on", "pressor_on",
    "lab_per_hr", "lab_distinct_itemid_per_hr", "chart_decision_per_hr",
    "consult_new_per_hr", "consult_palliative_flag",
] + z_cols
fc_windows[out_cols].to_parquet(OUTPUT_DIR / f"05_full_code_scored_{DATE_TAG}.parquet", index=False)

# ---------------------------------------------------------------------------
# 5. Distribution summary by group
# ---------------------------------------------------------------------------
grp_summary = fc_windows.groupby("group").palliative_score.agg(
    n="size",
    n_stays=lambda s: s.index.map(fc_windows.stay_id).nunique() if False else None,  # placeholder
    mean="mean", median="median", std="std",
    p25=lambda s: s.quantile(0.25),
    p75=lambda s: s.quantile(0.75),
    p90=lambda s: s.quantile(0.90),
    p99=lambda s: s.quantile(0.99),
).drop(columns="n_stays").round(3)
grp_summary["n_stays"] = fc_windows.groupby("group").stay_id.nunique()
print("\nDistribution by group:")
print(grp_summary)

# ---------------------------------------------------------------------------
# 6. Per-stay persistence metrics
# ---------------------------------------------------------------------------
p99 = fc_windows.palliative_score.quantile(0.99)
p90 = fc_windows.palliative_score.quantile(0.90)
print(f"\nOverall p90 of Full Code window scores: {p90:.3f}")
print(f"Overall p99 of Full Code window scores: {p99:.3f}")

stay_level = fc_windows.groupby(["stay_id", "group"]).agg(
    n_windows=("palliative_score", "size"),
    max_score=("palliative_score", "max"),
    mean_score=("palliative_score", "mean"),
    n_high_p90=("palliative_score", lambda s: int((s >= p90).sum())),
    n_high_p99=("palliative_score", lambda s: int((s >= p99).sum())),
).reset_index()
stay_level["frac_high_p90"] = stay_level.n_high_p90 / stay_level.n_windows

stay_level.to_parquet(OUTPUT_DIR / f"05_stay_level_scores_{DATE_TAG}.parquet", index=False)

print("\nPer-stay persistence metrics by group:")
print(stay_level.groupby("group").agg(
    n_stays=("stay_id", "size"),
    mean_max_score=("max_score", "mean"),
    p90_max_score=("max_score", lambda s: s.quantile(0.90)),
    frac_any_high_p90=("n_high_p90", lambda s: (s > 0).mean()),
    frac_any_high_p99=("n_high_p99", lambda s: (s > 0).mean()),
).round(3))

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
# Figure 1: Histogram overall + stacked by group
fig1, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(fc_windows.palliative_score.clip(-3, 5), bins=80, color="#4C72B0", alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=0.8, ls="--", label="severity-matched baseline")
ax.axvline(p90, color="#C44E52", lw=1.5, ls="--", label=f"p90 = {p90:.2f}")
ax.axvline(p99, color="#8B0000", lw=1.5, ls="--", label=f"p99 = {p99:.2f}")
ax.set_xlabel("Palliative-ness score")
ax.set_ylabel("Full Code windows")
ax.set_title(f"Distribution of palliative-ness scores\nacross all {len(fc_windows):,} Full Code windows")
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Density-compared across groups (normalized)
ax = axes[1]
colors = {"never_transition": "#4C72B0", "pre_dnr": "#DD8452", "pre_cmo": "#C44E52"}
for grp, color in colors.items():
    s = fc_windows[fc_windows.group == grp].palliative_score.clip(-3, 5)
    ax.hist(s, bins=60, density=True, alpha=0.45, label=f"{grp} (n={len(s):,})", color=color)
ax.axvline(0, color="black", lw=0.8, ls="--")
ax.set_xlabel("Palliative-ness score")
ax.set_ylabel("Density")
ax.set_title("Score distribution by downstream outcome group")
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig1_path = OUTPUT_DIR / f"05_score_histogram_{DATE_TAG}.png"
fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
print(f"Wrote {fig1_path}")

# Figure 2: Per-stay max score comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes2[0]
for grp, color in colors.items():
    s = stay_level[stay_level.group == grp].max_score
    ax.hist(s.clip(-2, 5), bins=50, density=True, alpha=0.45, label=f"{grp} (n={len(s):,})", color=color)
ax.axvline(p90, color="#C44E52", lw=1.2, ls="--", alpha=0.7, label=f"window p90 = {p90:.2f}")
ax.axvline(p99, color="#8B0000", lw=1.2, ls="--", alpha=0.7, label=f"window p99 = {p99:.2f}")
ax.set_xlabel("Per-stay MAX palliative-ness score")
ax.set_ylabel("Density")
ax.set_title("Stay-level peak score by group")
ax.legend(loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Persistence: fraction of windows with score ≥ p90, per stay
ax = axes2[1]
for grp, color in colors.items():
    s = stay_level[stay_level.group == grp].frac_high_p90
    ax.hist(s, bins=30, density=True, alpha=0.45, label=f"{grp} (n={len(s):,})", color=color)
ax.set_xlabel("Fraction of a stay's Full Code windows with score ≥ p90")
ax.set_ylabel("Density")
ax.set_title("Persistence of high-score windows per stay")
ax.legend(loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig2_path = OUTPUT_DIR / f"05_score_by_group_{DATE_TAG}.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"Wrote {fig2_path}")

# ---------------------------------------------------------------------------
# 8. Vitrine
# ---------------------------------------------------------------------------
def df_to_md(df):
    df = df.reset_index() if df.index.name is not None else df
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


summary_md = f"""# Step 05 — All Full Code windows scored

## Scored set
- **Full Code windows:** {len(fc_windows):,} (from {fc_windows.stay_id.nunique():,} unique stays)
- **Thresholds (computed across all Full Code windows):**
  - p90 = **{p90:.2f}** (pragmatic "high" threshold)
  - p99 = **{p99:.2f}** (extreme tail)

## Score distribution by downstream outcome group
{df_to_md(grp_summary)}

## Per-stay "peak" score by group
{df_to_md(fc_windows.groupby('group').agg(
    n_stays=('stay_id','nunique'),
    mean_score=('palliative_score','mean'),
    p90_score=('palliative_score', lambda s: round(s.quantile(0.9), 3))
).reset_index())}

## Per-stay persistence: fraction of Full Code windows at or above p90
{df_to_md(stay_level.groupby('group').agg(
    n_stays=('stay_id','size'),
    mean_max_score=('max_score', lambda s: round(s.mean(),3)),
    pct_any_high_p90=('n_high_p90', lambda s: round(100 * (s > 0).mean(), 1)),
    pct_any_high_p99=('n_high_p99', lambda s: round(100 * (s > 0).mean(), 1)),
).reset_index())}

## Interpretation
- **Never-transition group** defines the "baseline" distribution of scores in patients who never reach a formal care limit.
- **pre_dnr** and **pre_cmo** groups show score *within the Full Code phase* of patients who eventually transitioned. These are the within-cohort validators: if the method works, the pre-transition Full Code windows should show higher scores than never-transition, because they include the "silent de-escalation days" leading up to the formal order.
- The fraction of stays with ANY window above p90 is the simplest cross-group test: never_transition will be 10% by construction, but pre_dnr and pre_cmo should be enriched.
- **High-score windows in never-transition stays** are the primary candidates for "silent" Full Code de-escalation. Script 07 characterizes this population.
"""
show(summary_md, title="Step 05 — Full Code windows scored", study=STUDY, source="scripts/05_score_full_code_windows_20260411.py")
show(fig1, title="Step 05 — Score distribution", description="Left: overall histogram of palliative-ness scores across all Full Code windows. Right: density by downstream outcome group.", study=STUDY, source="scripts/05_score_full_code_windows_20260411.py")
show(fig2, title="Step 05 — Per-stay peak and persistence", description="Left: distribution of each stay's max score. Right: fraction of a stay's Full Code windows with score ≥ p90 (persistence).", study=STUDY, source="scripts/05_score_full_code_windows_20260411.py")

plt.close("all")
print("Done.")
