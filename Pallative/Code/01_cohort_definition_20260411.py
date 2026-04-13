"""
Script 01: Cohort Definition — Silent ICU De-escalation

Builds the eligible ICU-stay cohort for the "silent de-escalation" study.

Cohort (lax assumption, confirmed with researcher 2026-04-11):
  - Adult MIMIC-IV ICU stays, admission_age >= 18
  - ICU LOS >= 48 h
  - Stays with no code status ever charted are treated as IMPLICIT FULL CODE
    throughout. Justification: in-hospital mortality for no-cs-documented
    (13.0%) is within 2pp of explicit-Full-Code (11.0%), suggesting the two
    groups are clinically equivalent.

Four code-status patterns are labeled:
  - no_cs_documented : no Code Status chartevent ever
  - only_fullcode    : Full code charted, no DNR/CMO ever
  - reaches_dnr      : DNR ever but no CMO
  - reaches_cmo      : CMO ever (may also have DNR)

Outputs (to outputs/):
  - 01_cohort_20260411.parquet       : per-stay dataframe (primary)
  - 01_flowchart_20260411.csv        : inclusion flow counts
  - 01_cs_pattern_table_20260411.csv : breakdown of the 4 patterns
"""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from vitrine import section, show

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "path/to/mimiciv.duckdb"  # <-- UPDATE THIS TO YOUR DUCKDB PATH

con = duckdb.connect(DB_PATH, read_only=True)

section("Step 01: Cohort Definition", study=STUDY)

# ---------------------------------------------------------------------------
# 1. Flowchart counts
# ---------------------------------------------------------------------------
flow_sql = """
SELECT
    (SELECT COUNT(*) FROM mimiciv_icu.icustays) AS all_icu_stays,
    (SELECT COUNT(*)
       FROM mimiciv_icu.icustays i
       JOIN mimiciv_derived.icustay_detail icd USING (stay_id)
      WHERE icd.admission_age >= 18) AS adult_icu_stays,
    (SELECT COUNT(*)
       FROM mimiciv_icu.icustays i
       JOIN mimiciv_derived.icustay_detail icd USING (stay_id)
      WHERE icd.admission_age >= 18
        AND EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600.0 >= 48) AS adult_48h_stays
"""
flow = con.execute(flow_sql).df()

excluded_peds = int(flow.all_icu_stays[0] - flow.adult_icu_stays[0])
excluded_short = int(flow.adult_icu_stays[0] - flow.adult_48h_stays[0])
final_n = int(flow.adult_48h_stays[0])

flowchart = pd.DataFrame(
    [
        {"step": "All MIMIC-IV ICU stays", "n": int(flow.all_icu_stays[0]), "excluded": 0, "reason": ""},
        {"step": "Adult (age >= 18)", "n": int(flow.adult_icu_stays[0]), "excluded": excluded_peds, "reason": "pediatric"},
        {"step": "ICU LOS >= 48 h", "n": final_n, "excluded": excluded_short, "reason": "stay < 48 h"},
        {"step": "FINAL COHORT", "n": final_n, "excluded": 0, "reason": "(no-cs-documented treated as implicit Full Code)"},
    ]
)
flowchart.to_csv(OUTPUT_DIR / f"01_flowchart_{DATE_TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Main cohort pull
# ---------------------------------------------------------------------------
cohort_sql = """
WITH base AS (
    SELECT
        i.stay_id, i.subject_id, i.hadm_id,
        i.intime, i.outtime,
        EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600.0 AS los_hours,
        icd.admission_age, icd.gender, icd.race,
        icd.hospital_expire_flag, icd.dod,
        icd.icustay_seq, icd.first_icu_stay
    FROM mimiciv_icu.icustays i
    JOIN mimiciv_derived.icustay_detail icd USING (stay_id)
    WHERE icd.admission_age >= 18
      AND EXTRACT(EPOCH FROM (i.outtime - i.intime))/3600.0 >= 48
),
cs AS (
    SELECT
        ce.stay_id,
        COUNT(*) AS n_cs_events,
        MIN(CASE WHEN ce.value = 'Full code' THEN ce.charttime END) AS first_fullcode_time,
        MIN(CASE WHEN ce.value ILIKE '%DNR%' OR ce.value ILIKE '%DNAR%' THEN ce.charttime END) AS first_dnr_time,
        MIN(CASE WHEN ce.value ILIKE '%comfort%' THEN ce.charttime END) AS first_cmo_time
    FROM mimiciv_icu.chartevents ce
    WHERE ce.itemid IN (223758, 229784, 228687)
    GROUP BY ce.stay_id
),
all_icu AS (
    -- Full history of all ICU stays per subject (not restricted to cohort)
    -- so prior ICU days/count are complete.
    SELECT
        stay_id, subject_id, intime,
        EXTRACT(EPOCH FROM (outtime - intime))/3600.0/24.0 AS los_days
    FROM mimiciv_icu.icustays
),
prior_icu AS (
    SELECT
        stay_id,
        COALESCE(SUM(los_days) OVER (
            PARTITION BY subject_id ORDER BY intime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS prior_icu_days,
        COALESCE(COUNT(*) OVER (
            PARTITION BY subject_id ORDER BY intime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS prior_icu_stay_count
    FROM all_icu
),
ch AS (
    SELECT
        subject_id, hadm_id,
        charlson_comorbidity_index,
        dementia, metastatic_solid_tumor, severe_liver_disease,
        chronic_pulmonary_disease, congestive_heart_failure,
        renal_disease, malignant_cancer
    FROM mimiciv_derived.charlson
)
SELECT
    b.stay_id, b.subject_id, b.hadm_id,
    b.intime, b.outtime, b.los_hours,
    b.admission_age, b.gender, b.race,
    b.hospital_expire_flag, b.dod,
    b.icustay_seq, b.first_icu_stay,
    COALESCE(cs.n_cs_events, 0) AS n_cs_events,
    cs.first_fullcode_time, cs.first_dnr_time, cs.first_cmo_time,
    CASE
        WHEN cs.stay_id IS NULL THEN 'no_cs_documented'
        WHEN cs.first_cmo_time IS NOT NULL THEN 'reaches_cmo'
        WHEN cs.first_dnr_time IS NOT NULL THEN 'reaches_dnr'
        WHEN cs.first_fullcode_time IS NOT NULL THEN 'only_fullcode'
        ELSE 'other'
    END AS cs_pattern,
    prior_icu.prior_icu_days, prior_icu.prior_icu_stay_count,
    ch.charlson_comorbidity_index,
    ch.dementia, ch.metastatic_solid_tumor, ch.severe_liver_disease,
    ch.chronic_pulmonary_disease, ch.congestive_heart_failure,
    ch.renal_disease, ch.malignant_cancer
FROM base b
LEFT JOIN cs USING (stay_id)
LEFT JOIN prior_icu USING (stay_id)
LEFT JOIN ch ON ch.hadm_id = b.hadm_id
"""
cohort = con.execute(cohort_sql).df()
cohort.to_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet", index=False)

# ---------------------------------------------------------------------------
# 3. Code status pattern breakdown (descriptive)
# ---------------------------------------------------------------------------
pattern = (
    cohort.groupby("cs_pattern")
    .agg(
        n_stays=("stay_id", "size"),
        pct_inhosp_death=("hospital_expire_flag", lambda s: round(100 * s.mean(), 1)),
        mean_age=("admission_age", lambda s: round(s.mean(), 1)),
        mean_los_hours=("los_hours", lambda s: round(s.mean(), 1)),
        mean_charlson=("charlson_comorbidity_index", lambda s: round(s.mean(), 1)),
        mean_prior_icu_days=("prior_icu_days", lambda s: round(s.mean(), 1)),
        pct_female=("gender", lambda s: round(100 * (s == "F").mean(), 1)),
    )
    .reset_index()
    .sort_values("n_stays", ascending=False)
    .reset_index(drop=True)
)
pattern.to_csv(OUTPUT_DIR / f"01_cs_pattern_table_{DATE_TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Vitrine: flowchart + pattern table + summary
# ---------------------------------------------------------------------------
flow_md = f"""# Cohort flow

| Step | N | Excluded | Reason |
|---|---:|---:|---|
| All MIMIC-IV ICU stays | {int(flow.all_icu_stays[0]):,} | — | — |
| Adult (age ≥ 18) | {int(flow.adult_icu_stays[0]):,} | {excluded_peds:,} | pediatric |
| ICU LOS ≥ 48 h | {final_n:,} | {excluded_short:,} | stay < 48 h |
| **Final cohort** | **{final_n:,}** | — | no-cs-documented treated as implicit Full Code (lax assumption) |
"""
show(
    flow_md,
    title="Step 01 — Cohort flow",
    description="Inclusion flowchart. See outputs/01_flowchart_*.csv.",
    study=STUDY,
    source="scripts/01_cohort_definition_20260411.py",
)

show(
    pattern,
    title="Step 01 — Code status pattern breakdown",
    description=(
        "Four code-status patterns in the 48h-adult cohort. Mortality parity "
        "between 'no_cs_documented' (13%) and 'only_fullcode' (11%) supports "
        "treating the undocumented group as implicit Full Code."
    ),
    study=STUDY,
    source="scripts/01_cohort_definition_20260411.py",
)

def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


summary_md = f"""# Cohort summary

- **Final N (stays):** {len(cohort):,}
- **Unique subjects:** {cohort.subject_id.nunique():,}
- **Unique hospital admissions:** {cohort.hadm_id.nunique():,}
- **Mean age:** {cohort.admission_age.mean():.1f}
- **Female:** {(cohort.gender == 'F').mean() * 100:.1f}%
- **In-hospital mortality:** {cohort.hospital_expire_flag.mean() * 100:.1f}%
- **Mean ICU LOS (h):** {cohort.los_hours.mean():.1f}
- **Mean Charlson:** {cohort.charlson_comorbidity_index.mean():.1f}

## Distribution of code-status patterns
{df_to_md(pattern)}

## Hypothesis-relevant risk variables (kept as features, not adjusted for)
- Charlson Comorbidity Index (available for {cohort.charlson_comorbidity_index.notna().sum():,} stays)
- Cumulative prior ICU days across all admissions
- Prior ICU stay count
- Current ICU LOS (runtime at each 24-h window — computed in Script 02)
- Admission age

## Training set for the shift direction (next script)
Of stays that reach CMO in ICU, **205** have ≥24 h Full Code before + ≥6 h
after the first CMO order and will be used to define the within-patient
"rescue → palliative" shift direction in engagement-residual space.
"""
show(
    summary_md,
    title="Step 01 — Cohort summary",
    description="Headline numbers for the eligible cohort.",
    study=STUDY,
    source="scripts/01_cohort_definition_20260411.py",
)

con.close()
print(f"Cohort rows: {len(cohort):,}")
print(f"Wrote: {OUTPUT_DIR}/01_cohort_{DATE_TAG}.parquet")
print(f"Wrote: {OUTPUT_DIR}/01_flowchart_{DATE_TAG}.csv")
print(f"Wrote: {OUTPUT_DIR}/01_cs_pattern_table_{DATE_TAG}.csv")
