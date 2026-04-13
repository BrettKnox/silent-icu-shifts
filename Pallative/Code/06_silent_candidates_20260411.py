"""
Script 06: Silent De-escalation Candidate Identification
==========================================================

Identify ICU stays whose Full Code phase shows a low → moderately-elevated
palliative-score transition with non-improving SOFA — the "silent
de-escalation" profile.

Filters applied (stay-level, in order, tracked by equiflow):
  1. Eligible cohort: all Full Code stays with ≥3 scored windows
     (otherwise we can't observe baseline + 2-window persistence)
  2. Baseline (A1): at least one early-stay window (first 48 h) with
     palliative_score < 0.5
  3. Persistence (B1): at least 2 CONSECUTIVE windows with score ≥ 0.5,
     and these windows must occur AFTER the baseline window
  4. SOFA filter (C1-simple): SOFA during the first persistent-elevated
     window is ≥ SOFA during the baseline window
     (patient not getting clinically better on SOFA terms)

Outputs:
  - outputs/06_candidate_stays_20260411.parquet
  - outputs/06_consort_flow_20260411.png (equiflow flow diagram)
  - outputs/06_candidate_summary_20260411.csv
  - outputs/06_candidate_by_group_20260411.png
  - outputs/06_sample_trajectories_20260411.png
"""

import os
import pickle
import warnings
from datetime import date
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from equiflow import EquiFlow
from vitrine import section, show

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(all="ignore")

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

section("Step 06: Silent de-escalation candidates", study=STUDY)

# Thresholds (finalized 2026-04-11 after filter sweep).
# An elevated threshold of 0.5 produced 23% candidate rates across all groups
# with no between-group enrichment — not discriminating. Raised to 1.5, which
# is ~3x the day -2 pre-CMO signal and ~75% of the day -1 signal, gives 2.2x
# CMO enrichment over never_transition, and leaves ~1,200 never_transition
# candidates for characterization.
THRESH_ELEVATED = 1.5           # "meaningfully elevated" palliative score
THRESH_BASELINE = 0.5           # below this = "normal/rescue mode"
BASELINE_WINDOW_HOURS = 48.0    # look for baseline in first 48 h of scored windows
PERSISTENCE_WINDOWS = 2         # minimum consecutive elevated windows (48 h)

# ---------------------------------------------------------------------------
# 1. Load scored windows + cohort
# ---------------------------------------------------------------------------
fc = pd.read_parquet(OUTPUT_DIR / f"05_full_code_scored_{DATE_TAG}.parquet")
cohort = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
cohort = cohort[cohort.cs_pattern != "other"].reset_index(drop=True)

# Make sure windows are ordered within each stay
fc = fc.sort_values(["stay_id", "window_start"]).reset_index(drop=True)
fc["hours_since_intime"] = (
    fc.window_start - fc.stay_id.map(cohort.set_index("stay_id").intime)
).dt.total_seconds() / 3600.0

print(f"Scored Full Code windows: {len(fc):,} from {fc.stay_id.nunique():,} stays")

# Build per-stay table with n_windows, earliest baseline candidate, persistence
# run-length encoding, and the SOFA comparison.
def analyze_stay(g: pd.DataFrame) -> dict:
    g = g.sort_values("window_start").reset_index(drop=True)
    n_windows = len(g)
    out = {
        "stay_id": g.stay_id.iloc[0],
        "group": g.group.iloc[0],
        "n_windows": n_windows,
        "max_score": float(g.palliative_score.max()),
        "mean_score": float(g.palliative_score.mean()),
        "min_score": float(g.palliative_score.min()),
        "has_baseline_window": False,
        "baseline_idx": np.nan,
        "baseline_score": np.nan,
        "baseline_sofa": np.nan,
        "elevated_start_idx": np.nan,
        "elevated_start_score": np.nan,
        "elevated_start_sofa": np.nan,
        "persistent_elevated_run": 0,
        "sofa_non_improving_at_elevated": False,
        "is_candidate": False,
    }
    if n_windows < 3:
        return out

    # Baseline: any window with hours_since_intime <= BASELINE_WINDOW_HOURS and score < THRESH_BASELINE
    baseline_mask = (g.hours_since_intime <= BASELINE_WINDOW_HOURS) & (g.palliative_score < THRESH_BASELINE)
    if not baseline_mask.any():
        return out

    # Use the LAST such baseline window (latest low-score early window)
    baseline_idx = int(g[baseline_mask].index[-1])
    out["has_baseline_window"] = True
    out["baseline_idx"] = baseline_idx
    out["baseline_score"] = float(g.palliative_score.iloc[baseline_idx])
    out["baseline_sofa"] = float(g.sofa_max.iloc[baseline_idx]) if pd.notna(g.sofa_max.iloc[baseline_idx]) else np.nan

    # Persistence: find the first run of ≥ PERSISTENCE_WINDOWS consecutive
    # windows with score ≥ THRESH_ELEVATED, occurring AFTER baseline_idx
    after_baseline = g.iloc[baseline_idx + 1:].reset_index(drop=False).rename(columns={"index": "orig_idx"})
    if len(after_baseline) < PERSISTENCE_WINDOWS:
        return out

    elevated = (after_baseline.palliative_score >= THRESH_ELEVATED).values.astype(int)
    # Find first run of length ≥ PERSISTENCE_WINDOWS
    run = 0
    first_elev_start = None
    for i, e in enumerate(elevated):
        if e == 1:
            run += 1
            if run >= PERSISTENCE_WINDOWS and first_elev_start is None:
                first_elev_start = i - (PERSISTENCE_WINDOWS - 1)
                break
        else:
            run = 0
    if first_elev_start is None:
        return out

    elev_orig_idx = int(after_baseline.orig_idx.iloc[first_elev_start])
    out["elevated_start_idx"] = elev_orig_idx
    out["elevated_start_score"] = float(g.palliative_score.iloc[elev_orig_idx])
    out["elevated_start_sofa"] = float(g.sofa_max.iloc[elev_orig_idx]) if pd.notna(g.sofa_max.iloc[elev_orig_idx]) else np.nan

    # Measure run length from first_elev_start forward
    run_from_start = 0
    for e in elevated[first_elev_start:]:
        if e == 1:
            run_from_start += 1
        else:
            break
    out["persistent_elevated_run"] = int(run_from_start)

    # SOFA non-improving: elevated_start_sofa >= baseline_sofa
    if pd.notna(out["baseline_sofa"]) and pd.notna(out["elevated_start_sofa"]):
        out["sofa_non_improving_at_elevated"] = out["elevated_start_sofa"] >= out["baseline_sofa"]
    else:
        out["sofa_non_improving_at_elevated"] = False

    out["is_candidate"] = (
        out["has_baseline_window"]
        and out["persistent_elevated_run"] >= PERSISTENCE_WINDOWS
        and out["sofa_non_improving_at_elevated"]
    )
    return out


print("Analyzing each stay...")
per_stay_rows = [analyze_stay(g) for _, g in fc.groupby("stay_id", sort=False)]
per_stay = pd.DataFrame(per_stay_rows)

# Merge cohort context
per_stay = per_stay.merge(
    cohort[[
        "stay_id", "subject_id", "hadm_id", "admission_age", "gender", "race",
        "los_hours", "hospital_expire_flag", "charlson_comorbidity_index",
        "prior_icu_days", "prior_icu_stay_count", "cs_pattern",
    ]],
    on="stay_id", how="left",
)

# Pull marital_status from admissions (not in Script 01 cohort parquet)
DB_PATH = os.environ["MIMIC_DB_PATH"]  # export MIMIC_DB_PATH=/path/to/mimic_iv.duckdb
con = duckdb.connect(DB_PATH, read_only=True)
marital = con.execute(
    "SELECT hadm_id, marital_status FROM mimiciv_hosp.admissions"
).df()
con.close()
per_stay = per_stay.merge(marital, on="hadm_id", how="left")
per_stay["marital_status"] = per_stay.marital_status.fillna("UNKNOWN")
print(f"\nPer-stay rows: {len(per_stay):,}")
print(f"Candidates (all filters): {int(per_stay.is_candidate.sum()):,}")
print(f"Candidates by group:")
print(per_stay[per_stay.is_candidate].group.value_counts())

per_stay.to_parquet(OUTPUT_DIR / f"06_candidate_stays_{DATE_TAG}.parquet", index=False)

# ---------------------------------------------------------------------------
# 2. Build equiflow CONSORT diagram
# ---------------------------------------------------------------------------
# Start from ALL MIMIC-IV ICU stays and narrow progressively. Each exclusion
# step is clinically interpretable: <48h, had DNR/CMO, not enough data,
# not rescue-mode at admission, didn't shift, or got better.
# Only the never-transition cohort is tracked through this flow; pre-CMO
# and pre-DNR stays are used as positive controls in the separate
# candidate-rate-by-group plot.

con = duckdb.connect(DB_PATH, read_only=True)
all_icu = con.execute("""
    SELECT
        id.stay_id,
        id.subject_id,
        id.hadm_id,
        id.admission_age,
        id.gender,
        id.race,
        id.los_icu * 24.0 AS los_hours,
        id.hospital_expire_flag,
        id.icustay_seq,
        a.marital_status,
        c.charlson_comorbidity_index
    FROM mimiciv_derived.icustay_detail id
    LEFT JOIN mimiciv_hosp.admissions a ON id.hadm_id = a.hadm_id
    LEFT JOIN mimiciv_derived.charlson c ON id.hadm_id = c.hadm_id
""").df()
con.close()
print(f"\nAll MIMIC-IV ICU stays: {len(all_icu):,}")

# Load cs_pattern WITHOUT the "other" filter so DNI-only stays are tracked
cohort_all = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
all_icu = all_icu.merge(
    cohort_all[["stay_id", "cs_pattern"]], on="stay_id", how="left"
)

# Merge per_stay analysis flags (valid only for scored stays)
all_icu = all_icu.merge(
    per_stay[[
        "stay_id", "n_windows", "has_baseline_window",
        "persistent_elevated_run", "sofa_non_improving_at_elevated",
        "is_candidate",
    ]],
    on="stay_id", how="left",
)
all_icu["n_windows"] = all_icu.n_windows.fillna(0)
all_icu["persistent_elevated_run"] = all_icu.persistent_elevated_run.fillna(0)
for col in ["has_baseline_window", "sofa_non_improving_at_elevated", "is_candidate"]:
    all_icu[col] = all_icu[col].fillna(False).astype(bool)

# Subgroup categoricals
def _age_group(a):
    if pd.isna(a): return "unknown"
    if a < 45: return "18-44"
    if a < 65: return "45-64"
    if a < 75: return "65-74"
    if a < 85: return "75-84"
    return "85+"

def _collapse_race(r):
    if pd.isna(r): return "unknown"
    r = str(r).upper()
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic/Latino"
    if "BLACK" in r or "AFRICAN" in r: return "Black"
    if "ASIAN" in r: return "Asian"
    if "WHITE" in r or "PORTUGUESE" in r: return "White"
    if "NATIVE HAWAIIAN" in r or "PACIFIC ISLANDER" in r: return "Pacific Islander"
    if "AMERICAN INDIAN" in r or "ALASKA NATIVE" in r: return "AI/AN"
    return "Other/Unknown"

def _collapse_marital(m):
    if pd.isna(m) or m in ("UNKNOWN", ""): return "UNKNOWN"
    m = str(m).upper()
    if m in ("MARRIED", "SINGLE", "WIDOWED", "DIVORCED"): return m
    return "OTHER"

all_icu["long_los_21d"] = (all_icu.los_hours >= 21 * 24).map({True: "yes", False: "no"})
all_icu["multimorbid_charlson_ge5"] = np.where(
    all_icu.charlson_comorbidity_index.isna(), "unknown",
    np.where(all_icu.charlson_comorbidity_index >= 5, "yes", "no"),
)
all_icu["repeat_icu_user"] = (all_icu.icustay_seq >= 2).map({True: "yes", False: "no"})
all_icu["died_in_hospital"] = all_icu.hospital_expire_flag.map({1: "yes", 0: "no"}).fillna("unknown")
all_icu["age_group"] = all_icu.admission_age.map(_age_group)
all_icu["ethnicity"] = all_icu.race.map(_collapse_race)
all_icu["marital_status"] = all_icu.marital_status.map(_collapse_marital)

# Boolean keep masks for each CONSORT step
all_icu["keep_adult_48h"] = (all_icu.admission_age >= 18) & (all_icu.los_hours >= 48)
all_icu["keep_never_transition"] = all_icu.cs_pattern.isin(["no_cs_documented", "only_fullcode"])
all_icu["keep_n_windows"] = all_icu.n_windows >= 3
all_icu["keep_persistent"] = all_icu.persistent_elevated_run >= PERSISTENCE_WINDOWS

categorical = [
    "gender", "ethnicity", "age_group", "marital_status",
    "long_los_21d", "multimorbid_charlson_ge5", "repeat_icu_user",
    "died_in_hospital",
]

ef = EquiFlow(
    data=all_icu,
    initial_cohort_label="All MIMIC-IV ICU stays",
    categorical=categorical,
    nonnormal=None,
    label_suffix=False,
    thousands_sep=True,
    missingness=False,
)

ef.add_exclusion(
    keep=all_icu.keep_adult_48h,
    exclusion_reason="ICU LOS < 48 h",
    new_cohort_label="Adult stays ≥48 h",
)
ef.add_exclusion(
    keep=all_icu.keep_never_transition,
    exclusion_reason="Had formal DNR or CMO order",
    new_cohort_label="Never had formal DNR/CMO",
)
ef.add_exclusion(
    keep=all_icu.keep_n_windows,
    exclusion_reason="Not enough Full Code time to see trajectory",
    new_cohort_label="Enough ICU time for trajectory",
)
ef.add_exclusion(
    keep=all_icu.has_baseline_window,
    exclusion_reason="Already palliative-looking at admission",
    new_cohort_label="Rescue-mode at admission",
)
ef.add_exclusion(
    keep=all_icu.keep_persistent,
    exclusion_reason=f"No sustained shift ({PERSISTENCE_WINDOWS}+ consecutive windows ≥{THRESH_ELEVATED})",
    new_cohort_label="Baseline → persistent elevated shift",
)
ef.add_exclusion(
    keep=all_icu.sofa_non_improving_at_elevated,
    exclusion_reason="Severity improved during elevated period",
    new_cohort_label="Silent de-escalation candidates",
)

print("\nEquiflow cohort progression:")
try:
    flows_df = ef.view_table_flows()
    print(flows_df)
except Exception as e:
    print(f"(could not print flow table: {e})")

# Render flow diagram
consort_out_folder = str(OUTPUT_DIR)
consort_base = f"06_consort_flow_{DATE_TAG}"
try:
    ef.plot_flows(
        output_folder=consort_out_folder,
        output_file=consort_base,
        display_flow_diagram=False,
        plot_dists=True,     # show subgroup distributions at each node
        smds=True,           # standardized mean differences between nodes
        legend=True,
        legend_with_vars=True,
        box_width=3.2,
        box_height=1.2,
    )
    consort_path_pdf = OUTPUT_DIR / f"{consort_base}.pdf"
    consort_path_png = OUTPUT_DIR / f"{consort_base}.png"
    # Convert PDF to PNG for inline rendering in vitrine / Markdown
    if consort_path_pdf.exists():
        import subprocess
        subprocess.run(
            ["pdftoppm", "-png", "-r", "150", "-singlefile",
             str(consort_path_pdf), str(OUTPUT_DIR / consort_base)],
            check=False,
        )
    consort_path = consort_path_png if consort_path_png.exists() else consort_path_pdf
    print(f"Wrote CONSORT diagram: {consort_path}")
except Exception as e:
    print(f"equiflow plot_flows failed: {e}")
    consort_path = None

# ---------------------------------------------------------------------------
# 3. Candidate characterization (stratified by downstream group)
# ---------------------------------------------------------------------------
candidates = per_stay[per_stay.is_candidate].copy()

# Rates by group
denom = per_stay.groupby("group").size()
numer = candidates.groupby("group").size()
rates = pd.DataFrame({
    "group": denom.index,
    "n_total": denom.values,
    "n_candidates": [int(numer.get(g, 0)) for g in denom.index],
})
rates["pct_candidates"] = (rates.n_candidates / rates.n_total * 100).round(2)
print("\nCandidate rates by group:")
print(rates)
rates.to_csv(OUTPUT_DIR / f"06_candidate_summary_{DATE_TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Bar chart of candidate rate by group
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6.5, 4))
colors = {"never_transition": "#4C72B0", "pre_dnr": "#DD8452", "pre_cmo": "#C44E52"}
groups = rates.group.tolist()
bars = ax1.bar(range(len(groups)), rates.pct_candidates.values,
               color=[colors[g] for g in groups])
for i, r in rates.iterrows():
    ax1.text(i, r.pct_candidates, f"{r.pct_candidates:.1f}%\nn={r.n_candidates}/{r.n_total}",
             ha="center", va="bottom", fontsize=9)
ax1.set_xticks(range(len(groups)))
ax1.set_xticklabels(groups)
ax1.set_ylabel("% stays meeting candidate criteria")
ax1.set_title(f"Silent de-escalation candidates\n(low baseline → ≥{PERSISTENCE_WINDOWS} consec windows ≥ {THRESH_ELEVATED}, SOFA non-improving)")
ax1.set_ylim(0, rates.pct_candidates.max() * 1.35)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
fig1_path = OUTPUT_DIR / f"06_candidate_by_group_{DATE_TAG}.png"
fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
print(f"Wrote {fig1_path}")

# ---------------------------------------------------------------------------
# 5. Sample trajectories: 6 candidates (2 per group) with score + SOFA
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
rng = np.random.default_rng(20260411)
for row_i, grp in enumerate(["never_transition", "pre_dnr", "pre_cmo"]):
    sub = candidates[candidates.group == grp]
    if len(sub) == 0:
        for col_i in range(2):
            axes[row_i, col_i].set_visible(False)
        continue
    sample_ids = rng.choice(sub.stay_id.values, size=min(2, len(sub)), replace=False)
    for col_i, sid in enumerate(sample_ids):
        ax = axes[row_i, col_i]
        g = fc[fc.stay_id == sid].sort_values("window_start")
        ax2 = ax.twinx()
        ax.plot(g.hours_since_intime, g.palliative_score, "-o", color=colors[grp],
                lw=2, markersize=5, label="palliative score")
        ax2.plot(g.hours_since_intime, g.sofa_max, "--s", color="gray",
                 alpha=0.7, markersize=4, label="SOFA max")
        ax.axhline(THRESH_ELEVATED, color=colors[grp], lw=0.7, ls=":")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_xlabel("Hours since ICU intime")
        ax.set_ylabel("Palliative score", color=colors[grp])
        ax2.set_ylabel("SOFA max", color="gray")
        ax.set_title(f"{grp} — stay {sid}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
plt.tight_layout()
fig2_path = OUTPUT_DIR / f"06_sample_trajectories_{DATE_TAG}.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"Wrote {fig2_path}")

# ---------------------------------------------------------------------------
# 6. Vitrine summary
# ---------------------------------------------------------------------------
def df_to_md(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


summary_md = f"""# Step 06 — Silent de-escalation candidates

## Filter criteria
1. Stay has ≥3 scored Full Code windows
2. **Baseline:** at least one window in the first 48 h with palliative score < {THRESH_BASELINE}
3. **Persistence:** at least {PERSISTENCE_WINDOWS} consecutive windows with palliative score ≥ {THRESH_ELEVATED}, occurring after the baseline window
4. **SOFA non-improving:** SOFA_max at the first elevated window ≥ SOFA_max at the baseline window

## Candidate rates by group
{df_to_md(rates)}

## Expected direction
- `pre_cmo` and `pre_dnr` patients are POSITIVE CONTROLS: they by definition had a mental-model shift. Their Full Code phase before the formal order should match the candidate profile.
- `never_transition` candidates are the primary research interest: silent de-escalation in patients who never had a formal DNR/CMO written.

## CONSORT-style cohort flow (equiflow)
See below for the flow diagram image.
"""
show(summary_md, title="Step 06 — Silent de-escalation candidates", study=STUDY, source="scripts/06_silent_candidates_20260411.py")
show(fig1, title="Step 06 — Candidate rate by group", study=STUDY, source="scripts/06_silent_candidates_20260411.py")
show(fig2, title="Step 06 — Sample candidate trajectories", description="Two example candidates per group. Solid line = palliative score; dashed line = SOFA max.", study=STUDY, source="scripts/06_silent_candidates_20260411.py")

if consort_path and Path(consort_path).exists():
    from PIL import Image
    try:
        img = Image.open(consort_path)
        show(img, title="Step 06 — CONSORT cohort flow (equiflow)",
             description="Progressive exclusion from all Full Code stays to candidate stays.",
             study=STUDY, source="scripts/06_silent_candidates_20260411.py")
    except Exception:
        show(f"CONSORT flow diagram written to {consort_path}", title="Step 06 — CONSORT cohort flow", study=STUDY)

plt.close("all")
print("Done.")
