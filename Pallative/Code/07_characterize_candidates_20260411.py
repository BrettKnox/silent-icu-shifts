"""
Script 07: Characterize Silent De-escalation Candidates
=========================================================

Compare three groups of Full Code ICU stays on demographics, utilization,
comorbidity, and outcomes:

  1. never_transition candidates (n=1,217): never had a formal DNR/CMO, but
     matched our silent-de-escalation filter (baseline → persistent elevated
     palliative score with non-improving SOFA).
  2. never_transition non-candidates (n=41,678): the rest of never_transition,
     our reference population (Full Code stays with "normal" engagement).
  3. pre_cmo candidates (n=32): validation anchor — stays we KNOW eventually
     transitioned to CMO, whose Full Code phase matched the silent profile.

Outputs:
  - outputs/07_table1_20260411.csv
  - outputs/07_odds_ratios_20260411.csv
  - outputs/07_forest_plot_20260411.png
  - outputs/07_continuous_comparison_20260411.png
"""

import warnings
from datetime import date
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from vitrine import section, show


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the back
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(all="ignore")

STUDY = "silent-deescalation"
DATE_TAG = date.today().strftime("%Y%m%d")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = "PATH_TO_MIMIC_DB"  # <-- UPDATE THIS to your local MIMIC DuckDB path

section("Step 07: Characterize silent candidates", study=STUDY)

# ---------------------------------------------------------------------------
# 1. Load and merge data
# ---------------------------------------------------------------------------
stays = pd.read_parquet(OUTPUT_DIR / f"06_candidate_stays_{DATE_TAG}.parquet")
cohort = pd.read_parquet(OUTPUT_DIR / f"01_cohort_{DATE_TAG}.parquet")
windows = pd.read_parquet(OUTPUT_DIR / f"02_windows_{DATE_TAG}.parquet")

# Pull admissions.discharge_location (marital_status already in 06 parquet)
con = duckdb.connect(DB_PATH, read_only=True)
adm = con.execute(
    "SELECT hadm_id, discharge_location FROM mimiciv_hosp.admissions"
).df()
con.close()

# Ever-had-palliative-consult during the stay (any window)
palliative_ever = (
    windows.groupby("stay_id")["consult_palliative_flag"]
    .max()
    .rename("palliative_consult_ever")
    .reset_index()
)
palliative_ever["palliative_consult_ever"] = palliative_ever.palliative_consult_ever.astype(int)

s = stays.merge(adm, on="hadm_id", how="left") \
         .merge(palliative_ever, on="stay_id", how="left")
s["palliative_consult_ever"] = s.palliative_consult_ever.fillna(0).astype(int)
s["marital_status"] = s.marital_status.fillna("UNKNOWN")
s["discharge_location"] = s.discharge_location.fillna("UNKNOWN")

# ---------------------------------------------------------------------------
# 2. Derive categorical/binary variables
# ---------------------------------------------------------------------------
def age_group(a):
    if pd.isna(a): return "unknown"
    if a < 45: return "18-44"
    if a < 65: return "45-64"
    if a < 75: return "65-74"
    if a < 85: return "75-84"
    return "85+"

def collapse_race(r):
    if pd.isna(r): return "Other/Unknown"
    r = str(r).upper()
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic/Latino"
    if "BLACK" in r or "AFRICAN" in r: return "Black"
    if "ASIAN" in r: return "Asian"
    if "WHITE" in r or "PORTUGUESE" in r: return "White"
    return "Other/Unknown"

def collapse_marital(m):
    if pd.isna(m) or m == "": return "UNKNOWN"
    m = str(m).upper()
    if m in ("MARRIED", "SINGLE", "WIDOWED", "DIVORCED"): return m
    return "UNKNOWN"

def collapse_disch(d):
    if pd.isna(d) or d == "" or d == "UNKNOWN": return "UNKNOWN"
    d = str(d).upper()
    if d == "DIED": return "DIED"
    if "HOSPICE" in d: return "HOSPICE"
    if "HOME" in d: return "HOME"
    if "REHAB" in d or "SKILLED" in d or "CHRONIC" in d or "FACILITY" in d or "ACUTE HOSPITAL" in d:
        return "FACILITY"
    return "OTHER"

s["age_group"] = s.admission_age.map(age_group)
s["ethnicity"] = s.race.map(collapse_race)
s["marital_status"] = s.marital_status.map(collapse_marital)
s["discharge_bucket"] = s.discharge_location.map(collapse_disch)

s["long_los_21d"] = (s.los_hours >= 21 * 24).astype(int)
s["multimorbid_charlson_ge5"] = (s.charlson_comorbidity_index >= 5).astype(int)
s["repeat_icu_user"] = (s.prior_icu_stay_count >= 1).astype(int)
s["died_in_hospital"] = s.hospital_expire_flag.astype(int)

# ---------------------------------------------------------------------------
# 3. Build the three analysis groups
# ---------------------------------------------------------------------------
never_cand = s[(s.group == "never_transition") & (s.is_candidate)].copy()
never_noncand = s[(s.group == "never_transition") & (~s.is_candidate)].copy()
pre_cmo_cand = s[(s.group == "pre_cmo") & (s.is_candidate)].copy()

print(f"never_transition candidates: {len(never_cand):,}")
print(f"never_transition non-candidates: {len(never_noncand):,}")
print(f"pre_cmo candidates (validation anchor): {len(pre_cmo_cand):,}")

# ---------------------------------------------------------------------------
# 4. Table 1: characterization with pairwise tests
# ---------------------------------------------------------------------------
binary_vars = [
    ("long_los_21d", "Long LOS (≥21 days)"),
    ("multimorbid_charlson_ge5", "Multimorbid (Charlson ≥5)"),
    ("repeat_icu_user", "Repeat ICU user (≥1 prior stay)"),
    ("died_in_hospital", "Died in hospital"),
]
categorical_vars = [
    ("gender", "Gender"),
    ("ethnicity", "Ethnicity"),
    ("age_group", "Age group"),
    ("marital_status", "Marital status"),
    ("discharge_bucket", "Discharge destination"),
]
continuous_vars = [
    ("admission_age", "Age (years)"),
    ("los_hours", "ICU LOS (hours)"),
    ("charlson_comorbidity_index", "Charlson index"),
    ("prior_icu_days", "Prior ICU days (cumulative)"),
    ("prior_icu_stay_count", "Prior ICU stay count"),
    ("max_score", "Max palliative score in stay"),
]

table1_rows = []

def binary_test(a, b):
    """2x2 chi-square with OR + 95% CI. a, b are binary series (1/0)."""
    tab = np.array([
        [int((a == 1).sum()), int((a == 0).sum())],
        [int((b == 1).sum()), int((b == 0).sum())],
    ])
    _, p, _, _ = stats.chi2_contingency(tab, correction=False)
    # Haldane-Anscombe correction for zero cells
    tab_a = tab + 0.5
    odds_a = (tab_a[0, 0] * tab_a[1, 1]) / (tab_a[0, 1] * tab_a[1, 0])
    se = np.sqrt(1 / tab_a[0, 0] + 1 / tab_a[0, 1] + 1 / tab_a[1, 0] + 1 / tab_a[1, 1])
    lci = np.exp(np.log(odds_a) - 1.96 * se)
    uci = np.exp(np.log(odds_a) + 1.96 * se)
    return p, odds_a, lci, uci

def fmt_pct(series, val=1):
    n = int((series == val).sum())
    pct = 100 * n / len(series) if len(series) else 0
    return f"{n:,} ({pct:.1f}%)"

def fmt_median_iqr(series):
    s = series.dropna()
    if len(s) == 0: return "—"
    return f"{s.median():.1f} [{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}]"

# Binary vars
for var, label in binary_vars:
    row = {
        "variable": label,
        "type": "binary",
        "never_cand": fmt_pct(never_cand[var]),
        "never_noncand": fmt_pct(never_noncand[var]),
        "pre_cmo_cand": fmt_pct(pre_cmo_cand[var]),
    }
    p, or_, lci, uci = binary_test(never_cand[var], never_noncand[var])
    row["cand_vs_noncand_p"] = p
    row["cand_vs_noncand_OR"] = or_
    row["cand_vs_noncand_OR_ci"] = f"{or_:.2f} [{lci:.2f}–{uci:.2f}]"
    table1_rows.append(row)

# Continuous vars
for var, label in continuous_vars:
    a = never_cand[var].dropna()
    b = never_noncand[var].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else (np.nan, np.nan)
    row = {
        "variable": label,
        "type": "continuous",
        "never_cand": fmt_median_iqr(never_cand[var]),
        "never_noncand": fmt_median_iqr(never_noncand[var]),
        "pre_cmo_cand": fmt_median_iqr(pre_cmo_cand[var]),
        "cand_vs_noncand_p": p,
        "cand_vs_noncand_OR": np.nan,
        "cand_vs_noncand_OR_ci": "",
    }
    table1_rows.append(row)

# Multi-level categorical vars (chi-square)
for var, label in categorical_vars:
    levels = sorted(set(never_cand[var].dropna().unique()) | set(never_noncand[var].dropna().unique()))
    cand_counts = never_cand[var].value_counts()
    non_counts = never_noncand[var].value_counts()
    precmo_counts = pre_cmo_cand[var].value_counts()
    tab = np.array([
        [int(cand_counts.get(l, 0)) for l in levels],
        [int(non_counts.get(l, 0)) for l in levels],
    ])
    try:
        _, p_cat, _, _ = stats.chi2_contingency(tab, correction=False)
    except Exception:
        p_cat = np.nan
    table1_rows.append({
        "variable": label,
        "type": "categorical",
        "never_cand": "",
        "never_noncand": "",
        "pre_cmo_cand": "",
        "cand_vs_noncand_p": p_cat,
        "cand_vs_noncand_OR": np.nan,
        "cand_vs_noncand_OR_ci": "",
    })
    for lvl in levels:
        table1_rows.append({
            "variable": f"  {lvl}",
            "type": "",
            "never_cand": fmt_pct(never_cand[var] == lvl, val=True),
            "never_noncand": fmt_pct(never_noncand[var] == lvl, val=True),
            "pre_cmo_cand": fmt_pct(pre_cmo_cand[var] == lvl, val=True),
            "cand_vs_noncand_p": "",
            "cand_vs_noncand_OR": np.nan,
            "cand_vs_noncand_OR_ci": "",
        })

table1 = pd.DataFrame(table1_rows)

# BH-FDR correction on primary p-values
primary_pvals = pd.to_numeric(table1.cand_vs_noncand_p, errors="coerce")
mask_pval = primary_pvals.notna()
table1["cand_vs_noncand_p_fdr"] = np.nan
if mask_pval.any():
    pvals_corr = bh_fdr(primary_pvals[mask_pval].values)
    table1.loc[mask_pval, "cand_vs_noncand_p_fdr"] = pvals_corr

# Pretty p values
def fmt_p(p):
    if pd.isna(p) or p == "": return ""
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"
table1["p (raw)"] = table1.cand_vs_noncand_p.apply(fmt_p)
table1["p (FDR)"] = table1.cand_vs_noncand_p_fdr.apply(fmt_p)

out_cols = ["variable", "never_cand", "never_noncand", "pre_cmo_cand",
            "cand_vs_noncand_OR_ci", "p (raw)", "p (FDR)"]
table1_out = table1[out_cols].rename(columns={
    "variable": "Variable",
    "never_cand": f"Never-trans CAND (n={len(never_cand):,})",
    "never_noncand": f"Never-trans NON-cand (n={len(never_noncand):,})",
    "pre_cmo_cand": f"pre_CMO CAND (n={len(pre_cmo_cand):,})",
    "cand_vs_noncand_OR_ci": "OR [95% CI]",
})
table1_out.to_csv(OUTPUT_DIR / f"07_table1_{DATE_TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Forest plot of odds ratios — patient factors (top) vs outcomes (bottom)
# ---------------------------------------------------------------------------
# Each contrast is tagged with a category: "risk" = patient-level factor
# (pre-existing characteristic / process of care), "outcome" = discharge or
# mortality. Groups are plotted separately with outcomes at the bottom.
forest_contrasts = []

# Core binary risk factors (patient-level)
for var, label, cat in [
    ("long_los_21d", "Long LOS (≥21 days)", "risk"),
    ("multimorbid_charlson_ge5", "Multimorbid (Charlson ≥5)", "risk"),
    ("repeat_icu_user", "Repeat ICU user (≥1 prior stay)", "risk"),
    ("died_in_hospital", "Died in hospital", "outcome"),
]:
    forest_contrasts.append((
        label,
        never_cand[var].astype(int),
        never_noncand[var].astype(int),
        cat,
    ))

# Gender (patient factor)
forest_contrasts.append((
    "Male",
    (never_cand.gender == "M").astype(int),
    (never_noncand.gender == "M").astype(int),
    "risk",
))

# Age dichotomization: ≥75 vs <75
forest_contrasts.append((
    "Age ≥75 (vs <75)",
    (never_cand.admission_age >= 75).astype(int),
    (never_noncand.admission_age >= 75).astype(int),
    "risk",
))

# Ethnicity one-vs-rest (patient factor)
for ethn_val, ethn_label in [
    ("Black", "Black"),
    ("Hispanic/Latino", "Hispanic/Latino"),
    ("Asian", "Asian"),
]:
    forest_contrasts.append((
        ethn_label,
        (never_cand.ethnicity == ethn_val).astype(int),
        (never_noncand.ethnicity == ethn_val).astype(int),
        "risk",
    ))

# Marital status one-vs-rest (patient factor)
for m_val, m_label in [
    ("WIDOWED", "Widowed"),
    ("SINGLE", "Single"),
    ("DIVORCED", "Divorced"),
]:
    forest_contrasts.append((
        m_label,
        (never_cand.marital_status == m_val).astype(int),
        (never_noncand.marital_status == m_val).astype(int),
        "risk",
    ))

# Discharge destination (outcome; skip DIED — duplicates died_in_hospital)
for d_val, d_label in [
    ("HOSPICE", "Discharged to hospice"),
    ("HOME", "Discharged home"),
    ("FACILITY", "Discharged to facility"),
]:
    forest_contrasts.append((
        d_label,
        (never_cand.discharge_bucket == d_val).astype(int),
        (never_noncand.discharge_bucket == d_val).astype(int),
        "outcome",
    ))

# Run binary_test on every contrast
or_rows = []
for label, a, b, cat in forest_contrasts:
    p, or_, lci, uci = binary_test(a, b)
    or_rows.append({"variable": label, "OR": or_, "lci": lci, "uci": uci,
                    "p": p, "category": cat})
or_df = pd.DataFrame(or_rows)
or_df["p_fdr"] = bh_fdr(or_df.p.values)
or_df.to_csv(OUTPUT_DIR / f"07_odds_ratios_{DATE_TAG}.csv", index=False)

print("\nAll forest-plot contrasts:")
print(or_df.round(3).to_string(index=False))

# Filter to FDR-significant
forest_sig = or_df[or_df.p_fdr < 0.05].copy()

# Within each group, sort by OR ascending so the largest effect sits at the top
risk_rows = forest_sig[forest_sig.category == "risk"].sort_values(
    "OR", ascending=True
).reset_index(drop=True)
outcome_rows = forest_sig[forest_sig.category == "outcome"].sort_values(
    "OR", ascending=True
).reset_index(drop=True)
n_risk = len(risk_rows)
n_out = len(outcome_rows)
print(f"\nSignificant contrasts (FDR < 0.05): risk={n_risk}, outcome={n_out}")

# y positions: outcomes at bottom (y=0..n_out-1), gap, risk factors above
gap = 1.2
outcome_y = np.arange(n_out)
risk_y = np.arange(n_risk) + n_out + gap

# Plot
n_total_rows = n_out + n_risk
fig_height = max(4.5, 0.45 * n_total_rows + 2.0)
fig, ax = plt.subplots(figsize=(9, fig_height))

def _draw_row(yy, r):
    color = "#C44E52" if r.OR >= 1 else "#4C72B0"
    ax.plot([r.lci, r.uci], [yy, yy], color="#333", lw=2)
    ax.plot(r.OR, yy, "o", color=color, markersize=11,
            markeredgecolor="black", markeredgewidth=0.6)

for i, r in outcome_rows.iterrows():
    _draw_row(outcome_y[i], r)
for i, r in risk_rows.iterrows():
    _draw_row(risk_y[i], r)

ax.axvline(1, color="gray", ls="--", lw=1, alpha=0.8)

# Y-tick labels (combined)
all_y = np.concatenate([outcome_y, risk_y]) if n_out and n_risk else (outcome_y if n_out else risk_y)
all_labels = list(outcome_rows.variable) + list(risk_rows.variable)
ax.set_yticks(all_y)
ax.set_yticklabels(all_labels)

ax.set_xscale("log")
ax.set_xlabel("Odds ratio: never-transition CANDIDATE vs NON-CANDIDATE")
ax.set_title("Risk-factor enrichment in silent-de-escalation candidates\n"
             "(FDR-significant contrasts only)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate with OR, CI, FDR p
all_rows_combined = pd.concat([outcome_rows, risk_rows], ignore_index=True)
xmax = max(float(all_rows_combined.uci.max()), 1.5)
for yy, r in zip(all_y, all_rows_combined.itertuples()):
    ax.text(xmax * 1.15, yy,
            f"{r.OR:.2f} [{r.lci:.2f}–{r.uci:.2f}]  p$_{{FDR}}$={fmt_p(r.p_fdr)}",
            va="center", fontsize=9)
ax.set_xlim(min(0.4, float(all_rows_combined.lci.min()) * 0.8), xmax * 5.5)

# Set y-limits to leave room for group headers
y_top = (risk_y.max() if n_risk else outcome_y.max()) + 1.2
ax.set_ylim(-0.8, y_top)

# Group headers: x in axes-fraction (left edge), y in data coords
from matplotlib.transforms import blended_transform_factory
trans = blended_transform_factory(ax.transAxes, ax.transData)
if n_risk:
    ax.text(0.0, risk_y.max() + 0.6, "PATIENT FACTORS",
            transform=trans, fontsize=10, fontweight="bold",
            color="#333", ha="left", va="bottom")
if n_out:
    ax.text(0.0, outcome_y.max() + 0.6, "OUTCOMES",
            transform=trans, fontsize=10, fontweight="bold",
            color="#333", ha="left", va="bottom")

# Horizontal divider between groups (subtle)
if n_risk and n_out:
    divider_y = outcome_y.max() + gap / 2 + 0.2
    ax.axhline(divider_y, color="gray", ls="-", lw=0.4, alpha=0.25)

plt.tight_layout()
fig_forest = OUTPUT_DIR / f"07_forest_plot_{DATE_TAG}.png"
fig.savefig(fig_forest, dpi=150, bbox_inches="tight")
print(f"Wrote {fig_forest}")

# ---------------------------------------------------------------------------
# 6. Continuous comparison figure (violin-style boxplots)
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(2, 3, figsize=(13, 7))
cont_colors = ["#4C72B0", "#C44E52", "#55A868"]
group_order = ["never_noncand", "never_cand", "pre_cmo_cand"]
group_data = {
    "never_noncand": never_noncand,
    "never_cand": never_cand,
    "pre_cmo_cand": pre_cmo_cand,
}

for ax, (var, label) in zip(axes.flat, continuous_vars):
    data = [group_data[g][var].dropna().values for g in group_order]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], cont_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticklabels(["non-cand", "cand", "pre-CMO cand"], fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_cont = OUTPUT_DIR / f"07_continuous_comparison_{DATE_TAG}.png"
fig2.savefig(fig_cont, dpi=150, bbox_inches="tight")
print(f"Wrote {fig_cont}")

# ---------------------------------------------------------------------------
# 7. Save enriched stay table for reuse
# ---------------------------------------------------------------------------
s.to_parquet(OUTPUT_DIR / f"07_enriched_stays_{DATE_TAG}.parquet", index=False)

# ---------------------------------------------------------------------------
# 8. Vitrine
# ---------------------------------------------------------------------------
def df_to_md(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])

summary_md = f"""# Step 07 — Characterization of silent candidates

## Groups
- **never-trans CAND** (target): n = {len(never_cand):,} — Full Code stays that met the candidate filter but never had a formal DNR/CMO
- **never-trans NON-cand** (reference): n = {len(never_noncand):,} — the rest of the never_transition cohort
- **pre_CMO CAND** (positive control): n = {len(pre_cmo_cand):,} — stays that later became CMO and whose Full Code phase matched the silent profile

## Headline odds ratios (candidate vs non-candidate, within never_transition)
FDR-significant contrasts only — full contrast set is in `07_odds_ratios_{DATE_TAG}.csv`.

**Patient factors:**
{df_to_md(risk_rows.sort_values('OR', ascending=False)[['variable','OR','lci','uci','p_fdr']].assign(OR=lambda d: d.OR.round(2), lci=lambda d: d.lci.round(2), uci=lambda d: d.uci.round(2), p_fdr=lambda d: d.p_fdr.apply(fmt_p)))}

**Outcomes:**
{df_to_md(outcome_rows.sort_values('OR', ascending=False)[['variable','OR','lci','uci','p_fdr']].assign(OR=lambda d: d.OR.round(2), lci=lambda d: d.lci.round(2), uci=lambda d: d.uci.round(2), p_fdr=lambda d: d.p_fdr.apply(fmt_p)))}

## Full Table 1 (Variables × groups × p)
{df_to_md(table1_out.head(60))}

## Interpretation notes
- OR > 1 = over-represented among candidates (enriched risk factor).
- Statistical comparisons are never_cand vs never_noncand. FDR-corrected p-values control for the multiple-comparison set within this script.
- pre_CMO candidates column is NOT tested statistically (n too small) — it's shown side-by-side as an eyeball validation: if silent candidates in never_transition look demographically similar to pre_CMO candidates, that's supporting evidence we're capturing a related phenomenon.
"""

show(summary_md, title="Step 07 — Silent candidate characterization", study=STUDY, source="scripts/07_characterize_candidates_20260411.py")
show(fig, title="Step 07 — Forest plot of risk-factor odds ratios", description="OR for each binary risk factor: never-transition candidates vs non-candidates. Reference line at OR=1.", study=STUDY, source="scripts/07_characterize_candidates_20260411.py")
show(fig2, title="Step 07 — Continuous variable comparison", study=STUDY, source="scripts/07_characterize_candidates_20260411.py")

plt.close("all")
print("Done.")
