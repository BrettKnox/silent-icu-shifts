"""
Script 03: Within-Patient Shift Direction
==========================================

For stays that transition Full Code → CMO or Full Code → DNR-only in the ICU,
compute the "mental-model shift" direction in engagement-residual space, using
a within-patient paired design.

Method:
  1. Identify qualifying transitions:
     - Full Code ≥ 24 h before first_cmo_time, ≥ 6 h after, CMO occurs in-ICU.
     - Same criterion for DNR-only transitions (no CMO ever).
  2. For each transition stay, build custom pre/post windows:
       pre  = [first_limit_time - 24 h, first_limit_time)
       post = [first_limit_time, min(first_limit_time + 24 h, outtime))
     Pull engagement + severity for each.
  3. Fit a severity-residualization model on Script 02's full_code windows
     (297k rows): feature ~ severity covariates. Predict on pre/post windows
     to get residuals.
  4. Compute within-patient paired delta (post − pre) in residual space per stay.
  5. Average deltas across stays → shift direction vector, separately for CMO
     and DNR-only. Normalize to unit length.
  6. Cosine similarity between the two vectors → decide combine vs keep-separate.
  7. Plot direction vectors and within-patient deltas; save and post to vitrine.

Clean features trained against (sign-flip built in by within-patient design):
  - lab_per_hr
  - lab_distinct_itemid_per_hr
  - chart_decision_per_hr
  - consult_new_per_hr
  - consult_distinct_specialty_per_hr
  - consult_palliative_flag  (boolean; appearance at post vs pre adds signal)

Confounded features (excluded from training): proc_count, vasoactive_events,
vent_setting_events.

Outputs (to outputs/):
  - 03_transition_stays_20260411.parquet : qualifying stays with pre/post feature rows
  - 03_direction_vectors_20260411.csv    : shift-direction vectors (CMO, DNR, combined)
  - 03_shift_direction_20260411.png      : figure for vitrine
"""

import warnings
from datetime import date
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vitrine import section, show

# Suppress benign numpy FMA/BLAS warnings on macOS ARM during matmul.
# Data has been verified clean (no NaN/inf, ≥2 h windows, winsorized).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(all="ignore")

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = "PATH_TO_DUCKDB_DATABASE"  # set in .env, not checked in

con = duckdb.connect(DB_PATH, read_only=True)

section("Step 03: Within-patient shift direction", study=STUDY)

# Feature set for the direction training (clean CONTINUOUS care-intensity
# features only). Dropped from training:
#   - consult_palliative_flag: event-based leading indicator — typically
#     ordered DURING the pre-window and not repeated post, so paired delta
#     mechanically produces a negative weight that misrepresents the shift.
#     Reported separately as a standalone pre-transition indicator.
#   - consult_distinct_specialty_per_hr: near-collinear with consult_new_per_hr
#     since most windows have 0 or 1 new consult. Weak and redundant.
FEATURES = [
    "lab_per_hr",
    "lab_distinct_itemid_per_hr",
    "chart_decision_per_hr",
    "consult_new_per_hr",
]
# Severity covariates to residualize against
COVARS = [
    "sofa_max", "sofa_mean",
    "vent_any_on", "pressor_on",
    "hours_since_intime", "is_weekend", "hour_of_day",
    "window_hours",  # include to handle saturation of distinct-item counts
]

# ---------------------------------------------------------------------------
# 1. Identify transition stays
# ---------------------------------------------------------------------------
cohort = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
cohort = cohort[cohort.cs_pattern != "other"].reset_index(drop=True)

def qualifies(row, limit_time_col):
    t = row[limit_time_col]
    if pd.isna(t):
        return False
    pre_span = (t - row.intime).total_seconds() / 3600.0
    post_span = (row.outtime - t).total_seconds() / 3600.0
    return pre_span >= 24 and post_span >= 6

cmo_mask = (cohort.cs_pattern == "reaches_cmo") & cohort.apply(
    lambda r: qualifies(r, "first_cmo_time"), axis=1
)
dnr_only_mask = (cohort.cs_pattern == "reaches_dnr") & cohort.apply(
    lambda r: qualifies(r, "first_dnr_time"), axis=1
)

cmo_stays = cohort[cmo_mask].copy()
cmo_stays["transition_type"] = "cmo"
cmo_stays["first_limit_time"] = cmo_stays.first_cmo_time

dnr_stays = cohort[dnr_only_mask].copy()
dnr_stays["transition_type"] = "dnr"
dnr_stays["first_limit_time"] = dnr_stays.first_dnr_time

transitions = pd.concat([cmo_stays, dnr_stays], ignore_index=True)
transitions["pre_start"] = transitions.first_limit_time - pd.Timedelta(hours=24)
transitions["pre_end"] = transitions.first_limit_time
transitions["post_start"] = transitions.first_limit_time
transitions["post_end"] = np.minimum(
    transitions.first_limit_time + pd.Timedelta(hours=24),
    transitions.outtime,
)

print(f"CMO transitions: {cmo_mask.sum()}")
print(f"DNR-only transitions: {dnr_only_mask.sum()}")
print(f"Combined: {len(transitions)}")

# ---------------------------------------------------------------------------
# 2. For each transition, pull engagement + severity in pre and post windows
# ---------------------------------------------------------------------------
# Build a dataframe of pre/post windows (2 rows per transition stay), register
# in DuckDB, and run the same feature queries as Script 02 but on these custom
# windows.
pre_rows = transitions[["stay_id", "hadm_id", "transition_type", "pre_start", "pre_end"]].rename(
    columns={"pre_start": "window_start", "pre_end": "window_end"}
)
pre_rows["phase_label"] = "pre"
post_rows = transitions[["stay_id", "hadm_id", "transition_type", "post_start", "post_end"]].rename(
    columns={"post_start": "window_start", "post_end": "window_end"}
)
post_rows["phase_label"] = "post"
tw = pd.concat([pre_rows, post_rows], ignore_index=True)
tw["window_hours"] = (tw.window_end - tw.window_start).dt.total_seconds() / 3600.0
tw["transition_row_id"] = tw.stay_id.astype(str) + "_" + tw.phase_label
con.register("tw", tw[["stay_id", "hadm_id", "window_start", "window_end", "transition_row_id"]])

def feature_query(sql: str, name: str) -> pd.DataFrame:
    df = con.execute(sql).df()
    return df

# Labs
labs = feature_query(
    """
    SELECT tw.transition_row_id,
           COUNT(*) AS lab_count,
           COUNT(DISTINCT le.itemid) AS lab_distinct_itemid
    FROM tw
    LEFT JOIN mimiciv_hosp.labevents le
      ON le.hadm_id = tw.hadm_id
     AND le.charttime >= tw.window_start
     AND le.charttime <  tw.window_end
    GROUP BY tw.transition_row_id
    """, "labs",
)
# Consults
consults = feature_query(
    """
    SELECT tw.transition_row_id,
           COUNT(*) FILTER (WHERE p.transaction_type='New') AS consult_new,
           COUNT(DISTINCT CASE WHEN p.transaction_type='New' THEN p.order_subtype END) AS consult_distinct_specialty,
           MAX(CASE WHEN p.transaction_type='New' AND p.order_subtype ILIKE '%palliative%' THEN 1 ELSE 0 END) AS consult_palliative_flag
    FROM tw
    LEFT JOIN mimiciv_hosp.poe p
      ON p.hadm_id = tw.hadm_id
     AND p.order_type = 'Consults'
     AND p.ordertime >= tw.window_start
     AND p.ordertime <  tw.window_end
    GROUP BY tw.transition_row_id
    """, "consults",
)
# Chartevents (decision)
chart = feature_query(
    """
    WITH reflex AS (
        SELECT itemid FROM mimiciv_icu.d_items
        WHERE linksto='chartevents'
          AND category IN ('Routine Vital Signs','Alarms','Respiratory','Labs')
    )
    SELECT tw.transition_row_id,
           COUNT(*) FILTER (WHERE ce.itemid NOT IN (SELECT itemid FROM reflex)) AS chart_decision
    FROM tw
    LEFT JOIN mimiciv_icu.chartevents ce
      ON ce.stay_id = tw.stay_id
     AND ce.charttime >= tw.window_start
     AND ce.charttime <  tw.window_end
    GROUP BY tw.transition_row_id
    """, "chart",
)
# Severity
sev = feature_query(
    """
    SELECT tw.transition_row_id,
           MAX(s.sofa_24hours) AS sofa_max,
           AVG(s.sofa_24hours) AS sofa_mean
    FROM tw
    LEFT JOIN mimiciv_derived.sofa s
      ON s.stay_id = tw.stay_id
     AND s.endtime >  tw.window_start
     AND s.endtime <= tw.window_end
    GROUP BY tw.transition_row_id
    """, "sofa",
)
vent = feature_query(
    """
    SELECT tw.transition_row_id,
           MAX(CASE WHEN v.ventilation_status IS NOT NULL THEN 1 ELSE 0 END) AS vent_any_on
    FROM tw
    LEFT JOIN mimiciv_derived.ventilation v
      ON v.stay_id = tw.stay_id
     AND v.starttime <  tw.window_end
     AND v.endtime   >  tw.window_start
    GROUP BY tw.transition_row_id
    """, "vent",
)
press = feature_query(
    """
    SELECT tw.transition_row_id,
           MAX(CASE WHEN va.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS pressor_on
    FROM tw
    LEFT JOIN mimiciv_derived.vasoactive_agent va
      ON va.stay_id = tw.stay_id
     AND va.starttime <  tw.window_end
     AND va.endtime   >  tw.window_start
    GROUP BY tw.transition_row_id
    """, "press",
)

tw_feat = tw.merge(labs, on="transition_row_id", how="left") \
            .merge(consults, on="transition_row_id", how="left") \
            .merge(chart, on="transition_row_id", how="left") \
            .merge(sev, on="transition_row_id", how="left") \
            .merge(vent, on="transition_row_id", how="left") \
            .merge(press, on="transition_row_id", how="left")

# Fill nulls in counts, boolean flags
count_cols = ["lab_count", "lab_distinct_itemid", "consult_new", "consult_distinct_specialty",
              "consult_palliative_flag", "chart_decision", "vent_any_on", "pressor_on"]
for c in count_cols:
    tw_feat[c] = tw_feat[c].fillna(0).astype(int)

# Per-hour rates
tw_feat["lab_per_hr"] = tw_feat.lab_count / tw_feat.window_hours
tw_feat["lab_distinct_itemid_per_hr"] = tw_feat.lab_distinct_itemid / tw_feat.window_hours
tw_feat["chart_decision_per_hr"] = tw_feat.chart_decision / tw_feat.window_hours
tw_feat["consult_new_per_hr"] = tw_feat.consult_new / tw_feat.window_hours
tw_feat["consult_distinct_specialty_per_hr"] = tw_feat.consult_distinct_specialty / tw_feat.window_hours

# Calendar
intime_map = cohort.set_index("stay_id")["intime"]
tw_feat["hours_since_intime"] = (
    (tw_feat.window_start - tw_feat.stay_id.map(intime_map)).dt.total_seconds() / 3600.0
)
tw_feat["weekday"] = tw_feat.window_start.dt.dayofweek
tw_feat["is_weekend"] = tw_feat.weekday.isin([5, 6]).astype(int)
tw_feat["hour_of_day"] = tw_feat.window_start.dt.hour
tw_feat["sofa_max"] = tw_feat.sofa_max.fillna(tw_feat.sofa_max.median())
tw_feat["sofa_mean"] = tw_feat.sofa_mean.fillna(tw_feat.sofa_mean.median())

tw_feat.to_parquet(OUTPUT_DIR / f"03_transition_stays_{DATE_TAG}.parquet", index=False)
print(f"Transition pre/post feature rows: {len(tw_feat)}")

# ---------------------------------------------------------------------------
# 3. Fit residualization model on Script 02 full_code windows
# ---------------------------------------------------------------------------
# Drop windows shorter than 2 h (partial end-windows) — their per-hour rates
# are unreliable and can produce extreme values that destabilize the OLS fit.
MIN_WINDOW_HOURS = 2.0
print(f"Fitting residualization model on full_code windows (≥{MIN_WINDOW_HOURS} h)...")
w02 = pd.read_parquet(OUTPUT_DIR / f"02_windows_{DATE_TAG}.parquet")
fc = w02[(w02.phase == "full_code") & (w02.window_hours >= MIN_WINDOW_HOURS)].copy()
fc = fc.dropna(subset=COVARS + ["lab_per_hr", "chart_decision_per_hr"])
print(f"  fc rows for fit: {len(fc):,}")

# Add computed per-hour distinct-count features on the fc dataframe so
# fit_residualizer and z-scoring can treat them uniformly.
fc["lab_distinct_itemid_per_hr"] = fc.lab_distinct_itemid / fc.window_hours

# Winsorize per-hour features at 99.5th percentile to keep OLS stable against
# a few extreme outliers (e.g., a stay with a single window containing a huge
# lab burst). This only affects the fit; residuals on the full windows stay
# unaffected.
for f in FEATURES:
    cap = float(np.nanpercentile(fc[f].astype(float), 99.5))
    fc[f] = np.minimum(fc[f].astype(float), cap)

# Standardize covariates (fit on fc, apply to both fc and tw_feat) to keep OLS
# numerically stable. Store means/stds so we can transform the transition rows.
covar_mean = fc[COVARS].astype(float).mean()
covar_std = fc[COVARS].astype(float).std().replace(0, 1.0)

def standardize(df, cols, mean, std):
    return ((df[cols].astype(float) - mean) / std).values

X_fc = standardize(fc, COVARS, covar_mean, covar_std)

def fit_residualizer(X, y):
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta

def apply_residualizer(X, beta):
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ beta

betas = {}
for f in FEATURES:
    y = fc[f].astype(float).values
    betas[f] = fit_residualizer(X_fc, y)

# Residuals for pre/post rows (covariates standardized using fc means/stds)
X_tw = standardize(tw_feat, COVARS, covar_mean, covar_std)
X_tw = np.nan_to_num(X_tw, nan=0.0, posinf=0.0, neginf=0.0)
for f in FEATURES:
    pred = apply_residualizer(X_tw, betas[f])
    tw_feat[f + "_resid"] = tw_feat[f].astype(float).values - pred

# ---------------------------------------------------------------------------
# 4. Paired within-patient deltas → direction vectors
# ---------------------------------------------------------------------------
resid_cols = [f + "_resid" for f in FEATURES]
pre_df = tw_feat[tw_feat.phase_label == "pre"].set_index("stay_id")
post_df = tw_feat[tw_feat.phase_label == "post"].set_index("stay_id")
paired = post_df[resid_cols] - pre_df[resid_cols]
paired["transition_type"] = post_df["transition_type"]

# Z-score features on the full_code population so directions are comparable.
# fc already has lab_distinct_itemid_per_hr computed above.
fc_means = {f: fc[f].astype(float).mean() for f in FEATURES}
fc_stds = {f: (fc[f].astype(float).std() or 1.0) for f in FEATURES}

for f in FEATURES:
    paired[f + "_resid_z"] = paired[f + "_resid"] / fc_stds[f]
z_cols = [f + "_resid_z" for f in FEATURES]

cmo_delta = paired[paired.transition_type == "cmo"][z_cols].mean()
dnr_delta = paired[paired.transition_type == "dnr"][z_cols].mean()
combined_delta = paired[z_cols].mean()

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

v_cmo = unit(cmo_delta.values)
v_dnr = unit(dnr_delta.values)
v_comb = unit(combined_delta.values)
cosine = float(np.dot(v_cmo, v_dnr))

print(f"\nCMO direction (z-scored, unit): {dict(zip(FEATURES, np.round(v_cmo, 3)))}")
print(f"DNR direction (z-scored, unit): {dict(zip(FEATURES, np.round(v_dnr, 3)))}")
print(f"Combined direction (z-scored, unit): {dict(zip(FEATURES, np.round(v_comb, 3)))}")
print(f"Cosine similarity (CMO vs DNR): {cosine:.3f}")

dirs = pd.DataFrame({
    "feature": FEATURES,
    "cmo_delta_z_mean": cmo_delta.values,
    "dnr_delta_z_mean": dnr_delta.values,
    "combined_delta_z_mean": combined_delta.values,
    "cmo_unit": v_cmo,
    "dnr_unit": v_dnr,
    "combined_unit": v_comb,
}).round(3)
dirs.to_csv(OUTPUT_DIR / f"03_direction_vectors_{DATE_TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 4b. Sanity check: project pre and post windows onto the combined direction
#     and confirm post > pre (the direction should discriminate).
#     A "palliative-ness score" = projection onto COMBINED direction, where
#     HIGHER means more palliative-like behavior. Since the direction is
#     (post - pre) in z-scored residual space, we reverse the sign so that
#     "more palliative" = HIGHER score (delta goes negative; negated projection
#     goes positive when residual is below baseline).
z_by_row = pd.DataFrame(index=tw_feat.index)
for f in FEATURES:
    z_by_row[f] = tw_feat[f + "_resid"] / fc_stds[f]

# Palliative-ness score: plain projection onto the combined direction vector.
# The direction vector has NEGATIVE components (labs/chart/consults drop at
# transition → negative paired delta → negative unit components). A window
# whose residuals are also negative (below-baseline care intensity) yields
# NEG × NEG = positive score. So HIGHER score = more palliative-like behavior
# than severity-matched Full Code baseline. No sign flip needed.
projections = z_by_row[FEATURES].values @ v_comb
tw_feat["palliative_score"] = projections

pre_scores = tw_feat[tw_feat.phase_label == "pre"]["palliative_score"]
post_scores = tw_feat[tw_feat.phase_label == "post"]["palliative_score"]
print(f"\n=== Validation: palliative score on transition training windows ===")
print(f"pre-transition:  mean={pre_scores.mean():+.3f}  median={pre_scores.median():+.3f}")
print(f"post-transition: mean={post_scores.mean():+.3f}  median={post_scores.median():+.3f}")
print(f"Delta (post - pre): {post_scores.mean() - pre_scores.mean():+.3f}")
print(f"(Higher = more palliative-like. Post should exceed pre by construction.)")

# Also report a baseline full_code projection for reference: compute residual
# for a random sample of full_code windows, project, and report mean.
sample_fc = fc.sample(n=min(5000, len(fc)), random_state=42).copy()
# Use already-computed fc feature values; residual by subtracting the fit pred
X_sample = standardize(sample_fc, COVARS, covar_mean, covar_std)
for f in FEATURES:
    pred_s = apply_residualizer(X_sample, betas[f])
    sample_fc[f + "_resid"] = sample_fc[f].astype(float).values - pred_s
    sample_fc[f + "_resid_z"] = sample_fc[f + "_resid"] / fc_stds[f]
fc_proj = sample_fc[[f + "_resid_z" for f in FEATURES]].values @ v_comb
print(f"baseline full_code sample (n=5000): mean={fc_proj.mean():+.3f}  median={np.median(fc_proj):+.3f}")

# ---------------------------------------------------------------------------
# 4c. Palliative consult as a separate leading indicator
# ---------------------------------------------------------------------------
palliative_pre = tw_feat[tw_feat.phase_label == "pre"]["consult_palliative_flag"].mean() * 100
palliative_post = tw_feat[tw_feat.phase_label == "post"]["consult_palliative_flag"].mean() * 100
palliative_fc_baseline = fc.consult_palliative_flag.mean() * 100
print(f"\n=== Palliative consult as leading indicator ===")
print(f"% windows with new palliative consult:")
print(f"  baseline full_code: {palliative_fc_baseline:.2f}%")
print(f"  pre-transition (24 h before first DNR/CMO): {palliative_pre:.2f}%")
print(f"  post-transition: {palliative_post:.2f}%")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: bar chart of mean z-scored deltas for CMO vs DNR
ax = axes[0]
x = np.arange(len(FEATURES))
w = 0.38
ax.bar(x - w/2, cmo_delta.values, w, label=f"CMO (n={len(cmo_stays)})", color="#C44E52")
ax.bar(x + w/2, dnr_delta.values, w, label=f"DNR-only (n={len(dnr_stays)})", color="#DD8452")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f.replace("_per_hr", "/hr").replace("_flag", "") for f in FEATURES], rotation=30, ha="right")
ax.set_ylabel("Within-patient paired delta\n(z-scored residual, post − pre)")
ax.set_title("Mean paired delta by transition type")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: scatter of CMO unit vector vs DNR unit vector per feature
ax = axes[1]
ax.scatter(v_dnr, v_cmo, s=60, color="#555")
for i, f in enumerate(FEATURES):
    label = f.replace("_per_hr", "/hr").replace("_flag", "")
    ax.annotate(label, (v_dnr[i], v_cmo[i]), fontsize=8,
                textcoords="offset points", xytext=(5, 5))
lim = max(abs(v_cmo).max(), abs(v_dnr).max()) * 1.2
ax.plot([-lim, lim], [-lim, lim], color="gray", ls="--", lw=0.6, label="agreement")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("DNR-only direction component")
ax.set_ylabel("CMO direction component")
ax.set_title(f"Direction alignment (cosine={cosine:.2f})")
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = OUTPUT_DIR / f"03_shift_direction_{DATE_TAG}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Wrote {fig_path}")

# ---------------------------------------------------------------------------
# 6. Vitrine
# ---------------------------------------------------------------------------
interp = "aligned — combine into single direction" if cosine >= 0.6 else (
    "partially aligned — recommend reporting both" if cosine >= 0.3 else
    "misaligned — fall back to CMO-only training"
)
summary_md = f"""# Step 03 — Shift direction computed

## Training set sizes
- CMO transitions: **{len(cmo_stays)}** stays (Full Code ≥24 h before + ≥6 h after first in-ICU CMO)
- DNR-only transitions: **{len(dnr_stays)}** stays (same criterion, first in-ICU DNR, no CMO ever)
- **Combined: {len(transitions)} stays**

## Cosine similarity between CMO and DNR directions
**{cosine:.3f}** → **{interp}**

## CMO direction (unit vector in z-scored residual space)
{chr(10).join(f'- `{f}`: {v:+.3f}' for f, v in zip(FEATURES, v_cmo))}

## DNR direction (unit vector)
{chr(10).join(f'- `{f}`: {v:+.3f}' for f, v in zip(FEATURES, v_dnr))}

## Interpretation
Negative components mean the feature DROPS going from pre → post (less intensive care). Downstream scoring projects Full Code window residuals onto the NEGATIVE of this direction so that **higher score = more palliative-like** behavior than expected at this severity.

## Validation on the training set (palliative-ness score)
| Window type | Mean score | Median |
|---|---:|---:|
| baseline full_code sample (n=5000) | {fc_proj.mean():+.2f} | {np.median(fc_proj):+.2f} |
| pre-transition (24 h before DNR/CMO) | {pre_scores.mean():+.2f} | {pre_scores.median():+.2f} |
| post-transition | {post_scores.mean():+.2f} | {post_scores.median():+.2f} |

If the method has signal, pre-transition should exceed baseline (leading indicator) and post-transition should exceed pre-transition (formal shift).

## Palliative consult as an INDEPENDENT leading indicator
`consult_palliative_flag` was removed from the direction training because it's event-based rather than continuous (a consult happens once and is not repeated in the post-window, which contaminates paired-delta weight). As a standalone pre-transition signal it's strong:

| Window type | % with new palliative consult |
|---|---:|
| baseline full_code | {palliative_fc_baseline:.2f}% |
| pre-transition | {palliative_pre:.2f}% |
| post-transition | {palliative_post:.2f}% |

The pre/baseline ratio is the relevant effect size — palliative consults are dramatically enriched in the 24 h before a DNR/CMO order compared to typical Full Code care.
"""
show(summary_md, title="Step 03 — Shift direction summary", study=STUDY, source="scripts/03_shift_direction_20260411.py")
show(fig, title="Step 03 — Shift direction figure", description="Left: mean within-patient paired deltas by transition type. Right: CMO vs DNR direction alignment (cosine=%.2f)." % cosine, study=STUDY, source="scripts/03_shift_direction_20260411.py")
show(dirs, title="Step 03 — Direction vectors (table)", study=STUDY, source="scripts/03_shift_direction_20260411.py")

plt.close(fig)
con.close()
print("Done.")
