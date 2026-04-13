"""
Script 02: Engagement Features per 24-h ICU Window
===================================================

For each stay in the cohort, emits one row per 24-h window covering the ICU
stay. Columns include window metadata, code-status phase, severity covariates
(for residualizing in Script 04), and engagement features (the targets to
score).

Windows are anchored at ICU intime; the final window may be partial. Each
window row records `window_hours` so downstream scripts can normalize rates.

Drops the 26 `other` (DNI-only) stays per decision.

Outputs (to outputs/):
  - 02_windows_20260411.parquet : window-level feature matrix
  - 02_feature_coverage_20260411.csv : per-feature summary (non-null %, mean, p25/p50/p75)
"""

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from vitrine import section, show

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = "/path/to/mimiciv.duckdb"  # UPDATE THIS to your DuckDB path

con = duckdb.connect(DB_PATH, read_only=True)

section("Step 02: Engagement Features", study=STUDY)

# ---------------------------------------------------------------------------
# 1. Load cohort and build 24-h window grid
# ---------------------------------------------------------------------------
cohort = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
cohort = cohort[cohort.cs_pattern != "other"].reset_index(drop=True)
print(f"Cohort rows after dropping 'other': {len(cohort):,}")

# Build window grid: one row per (stay_id, window_idx)
cohort["n_windows"] = np.ceil(cohort.los_hours / 24.0).astype(int)
window_rows = []
for row in cohort.itertuples(index=False):
    for w in range(row.n_windows):
        w_start = row.intime + pd.Timedelta(hours=24 * w)
        w_end = min(row.intime + pd.Timedelta(hours=24 * (w + 1)), row.outtime)
        window_rows.append(
            {
                "stay_id": row.stay_id,
                "hadm_id": row.hadm_id,
                "subject_id": row.subject_id,
                "window_idx": w,
                "window_start": w_start,
                "window_end": w_end,
                "window_hours": (w_end - w_start).total_seconds() / 3600.0,
                "intime": row.intime,
                "first_fullcode_time": row.first_fullcode_time,
                "first_dnr_time": row.first_dnr_time,
                "first_cmo_time": row.first_cmo_time,
            }
        )

windows = pd.DataFrame(window_rows)
print(f"Generated {len(windows):,} windows across {windows.stay_id.nunique():,} stays")

# Derive phase label at window MIDPOINT
mid = windows.window_start + (windows.window_end - windows.window_start) / 2
is_cmo = windows.first_cmo_time.notna() & (mid >= windows.first_cmo_time)
is_dnr = windows.first_dnr_time.notna() & (mid >= windows.first_dnr_time) & ~is_cmo
windows["phase"] = np.select(
    [is_cmo, is_dnr],
    ["cmo_active", "dnr_active"],
    default="full_code",
)

# Calendar features at window start
windows["weekday"] = windows.window_start.dt.dayofweek  # 0=Mon
windows["is_weekend"] = windows.weekday.isin([5, 6]).astype(int)
windows["hour_of_day"] = windows.window_start.dt.hour
windows["hours_since_intime"] = (
    windows.window_start - windows.intime
).dt.total_seconds() / 3600.0

# Register in DuckDB for joins
con.register("windows", windows[[
    "stay_id", "hadm_id", "window_idx", "window_start", "window_end", "window_hours"
]])

print(f"Phase counts:\n{windows.phase.value_counts()}")

# ---------------------------------------------------------------------------
# 2. Feature helpers — run one LEFT JOIN per feature, merge by (stay_id, window_idx)
# ---------------------------------------------------------------------------
def add_feature(df: pd.DataFrame, sql: str, name: str) -> pd.DataFrame:
    feat = con.execute(sql).df()
    print(f"  {name}: {feat[feat.columns[-1]].notna().sum():,} windows populated")
    return df.merge(feat, on=["stay_id", "window_idx"], how="left")


# --- Labs (by hadm_id + time) ---
print("Labs...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           COUNT(*) AS lab_count,
           COUNT(DISTINCT le.itemid) AS lab_distinct_itemid
    FROM windows w
    LEFT JOIN mimiciv_hosp.labevents le
      ON le.hadm_id = w.hadm_id
     AND le.charttime >= w.window_start
     AND le.charttime <  w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "labs",
)

# --- POE consults (by hadm_id + ordertime), filter to New orders ---
print("Consults (POE)...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           COUNT(*) FILTER (WHERE p.transaction_type = 'New') AS consult_new,
           COUNT(DISTINCT CASE WHEN p.transaction_type='New' THEN p.order_subtype END) AS consult_distinct_specialty,
           MAX(CASE WHEN p.transaction_type='New' AND p.order_subtype ILIKE '%palliative%' THEN 1 ELSE 0 END) AS consult_palliative_flag
    FROM windows w
    LEFT JOIN mimiciv_hosp.poe p
      ON p.hadm_id = w.hadm_id
     AND p.order_type = 'Consults'
     AND p.ordertime >= w.window_start
     AND p.ordertime <  w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "consults",
)

# --- Chartevents total + excluding reflex/auto-captured + Labs (to avoid
#     double-counting labs already tracked via labevents). Verified 2026-04-11:
#     adding more categories to the exclude list does NOT sharpen the
#     full_code vs cmo_active contrast; the 4-category rule preserves signal
#     while removing the chartevents/labevents double-count.
print("Chartevents (may take a few minutes)...")
windows = add_feature(
    windows,
    """
    WITH reflex_items AS (
        SELECT itemid FROM mimiciv_icu.d_items
        WHERE linksto='chartevents'
          AND category IN ('Routine Vital Signs', 'Alarms', 'Respiratory', 'Labs')
    )
    SELECT w.stay_id, w.window_idx,
           COUNT(*) AS chart_total,
           COUNT(*) FILTER (WHERE ce.itemid NOT IN (SELECT itemid FROM reflex_items)) AS chart_decision
    FROM windows w
    LEFT JOIN mimiciv_icu.chartevents ce
      ON ce.stay_id = w.stay_id
     AND ce.charttime >= w.window_start
     AND ce.charttime <  w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "chartevents",
)

# --- Procedureevents ---
print("Procedures...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           COUNT(*) AS proc_count
    FROM windows w
    LEFT JOIN mimiciv_icu.procedureevents pe
      ON pe.stay_id = w.stay_id
     AND pe.starttime >= w.window_start
     AND pe.starttime <  w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "procedures",
)

# --- Ventilator setting changes (activity in vent settings table) ---
print("Ventilator settings...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           COUNT(*) AS vent_setting_events
    FROM windows w
    LEFT JOIN mimiciv_derived.ventilator_setting vs
      ON vs.stay_id = w.stay_id
     AND vs.charttime >= w.window_start
     AND vs.charttime <  w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "vent_settings",
)

# --- Vasoactive agent events (pressor titration proxy) ---
print("Vasoactive agents...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           COUNT(*) AS vasoactive_events
    FROM windows w
    LEFT JOIN mimiciv_derived.vasoactive_agent va
      ON va.stay_id = w.stay_id
     AND va.starttime >= w.window_start
     AND va.starttime <  w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "vasoactive",
)

# ---------------------------------------------------------------------------
# 3. Severity covariates (for residualizing in Script 04)
# ---------------------------------------------------------------------------
# SOFA aggregates from hourly sofa table
print("SOFA...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           MAX(s.sofa_24hours) AS sofa_max,
           AVG(s.sofa_24hours) AS sofa_mean
    FROM windows w
    LEFT JOIN mimiciv_derived.sofa s
      ON s.stay_id = w.stay_id
     AND s.endtime >  w.window_start
     AND s.endtime <= w.window_end
    GROUP BY w.stay_id, w.window_idx
    """,
    "sofa",
)

# Vent on (any overlap with a ventilation episode) — boolean
print("Vent on...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           MAX(CASE WHEN v.ventilation_status IN ('InvasiveVent','Trach') THEN 1 ELSE 0 END) AS vent_invasive_on,
           MAX(CASE WHEN v.ventilation_status IS NOT NULL THEN 1 ELSE 0 END) AS vent_any_on
    FROM windows w
    LEFT JOIN mimiciv_derived.ventilation v
      ON v.stay_id = w.stay_id
     AND v.starttime <  w.window_end
     AND v.endtime   >  w.window_start
    GROUP BY w.stay_id, w.window_idx
    """,
    "vent_on",
)

# Pressor on (any vasoactive_agent during window) — boolean
print("Pressor on...")
windows = add_feature(
    windows,
    """
    SELECT w.stay_id, w.window_idx,
           MAX(CASE WHEN va.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS pressor_on
    FROM windows w
    LEFT JOIN mimiciv_derived.vasoactive_agent va
      ON va.stay_id = w.stay_id
     AND va.starttime <  w.window_end
     AND va.endtime   >  w.window_start
    GROUP BY w.stay_id, w.window_idx
    """,
    "pressor_on",
)

# ---------------------------------------------------------------------------
# 4. Fill NaN counts → 0, save parquet
# ---------------------------------------------------------------------------
count_cols = [
    "lab_count", "lab_distinct_itemid",
    "consult_new", "consult_distinct_specialty", "consult_palliative_flag",
    "chart_total", "chart_decision",
    "proc_count",
    "vent_setting_events",
    "vasoactive_events",
    "vent_invasive_on", "vent_any_on", "pressor_on",
]
for c in count_cols:
    if c in windows.columns:
        windows[c] = windows[c].fillna(0).astype(int)

# Normalize to per-hour rates so partial end-windows don't bias downstream
windows["lab_per_hr"] = windows.lab_count / windows.window_hours
windows["chart_total_per_hr"] = windows.chart_total / windows.window_hours
windows["chart_decision_per_hr"] = windows.chart_decision / windows.window_hours
windows["proc_per_hr"] = windows.proc_count / windows.window_hours
windows["vent_setting_per_hr"] = windows.vent_setting_events / windows.window_hours
windows["vasoactive_per_hr"] = windows.vasoactive_events / windows.window_hours
windows["consult_new_per_hr"] = windows.consult_new / windows.window_hours

out_path = OUTPUT_DIR / f"02_windows_{DATE_TAG}.parquet"
windows.to_parquet(out_path, index=False)
print(f"\nWrote {len(windows):,} windows × {len(windows.columns)} cols → {out_path}")

# ---------------------------------------------------------------------------
# 5. Feature coverage summary
# ---------------------------------------------------------------------------
feature_cols = [
    "lab_count", "lab_per_hr", "lab_distinct_itemid",
    "chart_total", "chart_decision", "chart_decision_per_hr",
    "consult_new", "consult_distinct_specialty", "consult_palliative_flag",
    "proc_count", "vent_setting_events", "vasoactive_events",
    "sofa_max", "sofa_mean", "vent_any_on", "pressor_on",
]
summary = windows[feature_cols].describe(percentiles=[0.25, 0.5, 0.75]).T.round(2)
summary = summary.reset_index().rename(columns={"index": "feature"})
summary.to_csv(OUTPUT_DIR / f"02_feature_coverage_{DATE_TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Post to vitrine
# ---------------------------------------------------------------------------
def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


phase_counts = windows.phase.value_counts().to_dict()
overview_md = f"""# Step 02 — Engagement features built

## Window counts
- **Total windows:** {len(windows):,}
- **Unique stays:** {windows.stay_id.nunique():,}
- **Median windows per stay:** {windows.groupby('stay_id').size().median():.0f}
- **P90 windows per stay:** {windows.groupby('stay_id').size().quantile(0.9):.0f}

## Phase distribution (window-level)
- `full_code`: {phase_counts.get('full_code', 0):,}
- `dnr_active`: {phase_counts.get('dnr_active', 0):,}
- `cmo_active`: {phase_counts.get('cmo_active', 0):,}

## Feature categories extracted
| Category | Features |
|---|---|
| Labs | `lab_count`, `lab_per_hr`, `lab_distinct_itemid` |
| Consults (POE) | `consult_new`, `consult_distinct_specialty`, `consult_palliative_flag` |
| Charting | `chart_total`, `chart_decision` (excludes Routine Vital Signs, Alarms, Respiratory categories) |
| Interventions | `proc_count`, `vent_setting_events`, `vasoactive_events` |
| Severity (covariates) | `sofa_max`, `sofa_mean`, `vent_any_on`, `vent_invasive_on`, `pressor_on` |
| Time/calendar | `hours_since_intime`, `weekday`, `is_weekend`, `hour_of_day`, `window_hours` |

All count-like features are available both as raw counts and as per-hour rates
(e.g. `lab_per_hr`) so partial end-windows don't bias downstream analysis.
"""
show(overview_md, title="Step 02 — Engagement feature build complete", study=STUDY, source="scripts/02_engagement_features_20260411.py")

show(
    summary,
    title="Step 02 — Feature coverage & distributions",
    description="Per-feature summary statistics across all windows.",
    study=STUDY,
    source="scripts/02_engagement_features_20260411.py",
)

con.close()
print("Done.")
