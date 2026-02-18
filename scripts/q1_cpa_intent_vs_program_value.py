"""
Q1 Analysis: CPA License Intent (Q8) vs. MAcc Program Worth It (Q53)

Research Question:
    Is there a relationship between students planning to obtain their CPA
    license (Q8) and whether they felt the MAcc program was worth it (Q53)?

Usage:
    python scripts/q1_cpa_intent_vs_program_value.py

Environment Variables:
    DATA_PATH   Path to the Qualtrics-exported XLSX file.
                Default: data/Grad Program Exit Survey Data 2024.xlsx
"""

import json
import math
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("data") / "Grad Program Exit Survey Data 2024.xlsx"
REPORTS_DIR = Path("reports")

# Keywords used when the exact column code is not found
Q8_KEYWORDS = ["cpa", "license", "obtain", "planning on obtaining"]
Q53_KEYWORDS = ["worth", "program worth", "money"]

# Canonical label mapping for Q8 (CPA intent)
Q8_LABEL_MAP = {
    "yes": "Yes / Plan to get CPA",
    "no": "No / Do not plan",
    "maybe": "Unsure",
}
Q8_ORDER = ["Yes / Plan to get CPA", "Unsure", "No / Do not plan"]

# Likert ordered categories (used if Q53 matches this pattern)
LIKERT_LABELS = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]

# Canonical mapping for a simple Yes/No/Maybe Q53
Q53_SIMPLE_MAP = {
    "yes": "Yes",
    "no": "No",
    "maybe": "Maybe",
}
Q53_SIMPLE_ORDER = ["Yes", "Maybe", "No"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def _find_column(df: pd.DataFrame, code: str, keywords: list[str]) -> str:
    """
    Resolve a column name in *df* for a given question code/keywords.

    Resolution order:
      1. Exact match on *code* (e.g., "Q8").
      2. Columns whose name starts with *code* (e.g., "Q8_1").
      3. Columns whose name contains *code* as a token.
      4. Columns whose name contains any keyword from *keywords* (case-insensitive).

    Returns the unique matched column name or fails loudly.
    """
    cols = list(df.columns)

    # 1. Exact match
    if code in cols:
        print(f"  [{code}] Resolved via exact match: '{code}'")
        return code

    # 2. Starts with code
    starts = [c for c in cols if str(c).startswith(code)]
    if len(starts) == 1:
        print(f"  [{code}] Resolved via prefix match: '{starts[0]}'")
        return starts[0]
    if len(starts) > 1:
        _fail(
            f"Ambiguous columns starting with '{code}': {starts}\n"
            "Set DATA_PATH to a file with unique column names or fix the data."
        )

    # 3. Contains code as a word token
    token = [c for c in cols if code in str(c).split("_")]
    if len(token) == 1:
        print(f"  [{code}] Resolved via token match: '{token[0]}'")
        return token[0]
    if len(token) > 1:
        _fail(f"Ambiguous token matches for '{code}': {token}")

    # 4. Keyword search in header text
    kw_matches = [
        c for c in cols
        if any(kw.lower() in str(c).lower() for kw in keywords)
    ]
    if len(kw_matches) == 1:
        print(f"  [{code}] Resolved via keyword match: '{kw_matches[0]}'")
        return kw_matches[0]
    if len(kw_matches) > 1:
        _fail(
            f"Ambiguous keyword matches for '{code}' "
            f"(keywords={keywords}): {kw_matches}"
        )

    _fail(
        f"Cannot find column for '{code}'. "
        f"Columns available: {cols}"
    )


def _load_qualtrics_xlsx(path: Path) -> pd.DataFrame:
    """
    Load a Qualtrics XLSX export.

    Qualtrics exports have three header rows:
      Row 0: Question codes  (Q8, Q53, ...)
      Row 1: Question text
      Row 2: ImportId metadata

    We use row 0 as the column names and skip rows 1 and 2.
    """
    # Peek at row count to choose loading strategy
    raw = pd.read_excel(path, header=None, nrows=4)

    # Detect whether row 1 contains question text and row 2 contains ImportId
    row1_sample = str(raw.iloc[1, 0]) if len(raw) > 1 else ""
    row2_sample = str(raw.iloc[2, 0]) if len(raw) > 2 else ""

    if "ImportId" in row2_sample:
        # Standard Qualtrics 3-row header
        df = pd.read_excel(path, header=0, skiprows=[1, 2])
        print("  Detected standard Qualtrics 3-row header format.")
    elif row1_sample and row1_sample != "nan":
        # Possibly a 2-row header (codes + text); skip row 1
        df = pd.read_excel(path, header=0, skiprows=[1])
        print("  Detected 2-row header format (codes + text).")
    else:
        df = pd.read_excel(path, header=0)
        print("  Detected single-row header format.")

    return df


def _normalize_q8(series: pd.Series) -> pd.Series:
    """Map raw Q8 values to canonical CPA-intent labels."""
    def _map(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        if key in Q8_LABEL_MAP:
            return Q8_LABEL_MAP[key]
        # Numeric codes sometimes used
        if key in ("1",):
            return "Yes / Plan to get CPA"
        if key in ("2",):
            return "No / Do not plan"
        if key in ("3",):
            return "Unsure"
        # Partial matches
        if "yes" in key:
            return "Yes / Plan to get CPA"
        if "no" in key:
            return "No / Do not plan"
        if "maybe" in key or "unsure" in key or "not sure" in key:
            return "Unsure"
        # Return original capitalised if unrecognised
        return str(val).strip().title()

    return series.apply(_map)


def _detect_q53_type(series: pd.Series) -> str:
    """
    Determine whether Q53 looks like Likert or simple Yes/No/Maybe.
    Returns 'likert' or 'simple'.
    """
    vals = set(series.dropna().str.strip().str.lower().unique())
    likert_keys = {v.lower() for v in LIKERT_LABELS}
    if vals.issubset(likert_keys) and len(vals) > 0:
        return "likert"
    simple_keys = {"yes", "no", "maybe"}
    if vals.issubset(simple_keys) and len(vals) > 0:
        return "simple"
    return "simple"  # default: treat as categorical


def _normalize_q53(series: pd.Series, q53_type: str) -> pd.Series:
    """Map raw Q53 values to canonical labels."""
    def _map_likert(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        for label in LIKERT_LABELS:
            if key == label.lower():
                return label
        # Numeric 1-5
        numeric_map = {str(i + 1): LIKERT_LABELS[i] for i in range(5)}
        if key in numeric_map:
            return numeric_map[key]
        return str(val).strip().title()

    def _map_simple(val):
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        if key in Q53_SIMPLE_MAP:
            return Q53_SIMPLE_MAP[key]
        if "yes" in key:
            return "Yes"
        if "no" in key:
            return "No"
        if "maybe" in key:
            return "Maybe"
        return str(val).strip().title()

    if q53_type == "likert":
        return series.apply(_map_likert)
    return series.apply(_map_simple)


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Compute Cramér's V effect size."""
    return math.sqrt(chi2 / (n * (min(r, c) - 1)))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(data_path: Path) -> None:
    print("=" * 65)
    print("Q1: CPA Intent (Q8) vs. MAcc Worth It (Q53)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading data from: {data_path}")
    if not data_path.exists():
        _fail(
            f"Data file not found: {data_path}\n"
            "Set the DATA_PATH environment variable to the correct path."
        )

    df_raw = _load_qualtrics_xlsx(data_path)
    print(f"  Raw shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    # ------------------------------------------------------------------
    # 2. Resolve columns for Q8 and Q53
    # ------------------------------------------------------------------
    print("\n[2/6] Resolving columns ...")
    col_q8 = _find_column(df_raw, "Q8", Q8_KEYWORDS)
    col_q53 = _find_column(df_raw, "Q53", Q53_KEYWORDS)

    # ------------------------------------------------------------------
    # 3. Extract & clean
    # ------------------------------------------------------------------
    print("\n[3/6] Extracting and cleaning variables ...")

    work = df_raw[[col_q8, col_q53]].copy()
    work.columns = ["Q8_raw", "Q53_raw"]

    n_before = len(work)

    # Drop rows where either is missing/blank
    work = work.replace(r"^\s*$", np.nan, regex=True)
    work = work.dropna(subset=["Q8_raw", "Q53_raw"])
    n_after = len(work)
    print(f"  Dropped {n_before - n_after} rows with missing Q8 or Q53.")
    print(f"  Sample size after filtering: {n_after}")

    # Detect Q53 response type
    q53_type = _detect_q53_type(work["Q53_raw"].astype(str))
    print(f"  Q53 response type detected: '{q53_type}'")

    # Normalize
    work["Q8"] = _normalize_q8(work["Q8_raw"])
    work["Q53"] = _normalize_q53(work["Q53_raw"].astype(str), q53_type)

    # Drop any rows that became NaN after mapping (shouldn't happen, but safe)
    work = work.dropna(subset=["Q8", "Q53"])
    n_final = len(work)
    if n_final < n_after:
        print(f"  Dropped {n_after - n_final} rows with unrecognised values.")
    print(f"  Final sample size: {n_final}")

    # Determine display order
    q8_order = [c for c in Q8_ORDER if c in work["Q8"].unique()]
    q8_other = [c for c in work["Q8"].unique() if c not in Q8_ORDER]
    q8_order = q8_order + q8_other

    if q53_type == "likert":
        q53_order = [c for c in LIKERT_LABELS if c in work["Q53"].unique()]
    else:
        q53_order = [c for c in Q53_SIMPLE_ORDER if c in work["Q53"].unique()]
        q53_other = [c for c in work["Q53"].unique() if c not in Q53_SIMPLE_ORDER]
        q53_order = q53_order + q53_other

    # Apply CategoricalDtype for ordered display
    work["Q8"] = pd.Categorical(work["Q8"], categories=q8_order, ordered=True)
    work["Q53"] = pd.Categorical(work["Q53"], categories=q53_order, ordered=True)

    # Print distributions
    print("\n  Q8 (CPA Intent) distribution:")
    for label, cnt in work["Q8"].value_counts(sort=False).items():
        pct = 100 * cnt / n_final
        print(f"    {label:<30} {cnt:>4}  ({pct:5.1f}%)")

    print("\n  Q53 (Program Worth It) distribution:")
    for label, cnt in work["Q53"].value_counts(sort=False).items():
        pct = 100 * cnt / n_final
        print(f"    {label:<30} {cnt:>4}  ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # 4. Contingency table + percentages
    # ------------------------------------------------------------------
    print("\n[4/6] Computing contingency tables ...")

    ct_counts = pd.crosstab(
        work["Q8"], work["Q53"],
        rownames=["Q8 (CPA Intent)"],
        colnames=["Q53 (Worth It)"],
    )

    # Row-wise % (within Q8 groups)
    ct_row_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100

    # Column-wise % (within Q53 groups)
    ct_col_pct = ct_counts.div(ct_counts.sum(axis=0), axis=1) * 100

    print("\n  Contingency table (counts):")
    print(ct_counts.to_string())

    print("\n  Row-wise % (within each Q8 group):")
    print(ct_row_pct.round(1).to_string())

    print("\n  Column-wise % (within each Q53 group):")
    print(ct_col_pct.round(1).to_string())

    # ------------------------------------------------------------------
    # 5. Statistical tests
    # ------------------------------------------------------------------
    print("\n[5/6] Running statistical tests ...")

    chi2_stat, p_val, dof, expected = chi2_contingency(ct_counts)
    n_total = ct_counts.values.sum()
    r, c = ct_counts.shape
    v = _cramers_v(chi2_stat, n_total, r, c)

    print(f"  Chi-square statistic : {chi2_stat:.4f}")
    print(f"  Degrees of freedom   : {dof}")
    print(f"  p-value              : {p_val:.4f}")
    print(f"  Cramér's V           : {v:.4f}")

    # Significance interpretation
    if p_val < 0.001:
        sig_note = "p < 0.001 (highly significant)"
    elif p_val < 0.01:
        sig_note = "p < 0.01 (significant)"
    elif p_val < 0.05:
        sig_note = "p < 0.05 (significant)"
    else:
        sig_note = "p ≥ 0.05 (not significant at α=0.05)"

    # Effect size interpretation (Cramér's V for 2xK or 3xK tables)
    if v < 0.1:
        effect_note = "negligible"
    elif v < 0.3:
        effect_note = "small"
    elif v < 0.5:
        effect_note = "medium"
    else:
        effect_note = "large"

    print(f"  Interpretation       : {sig_note}")
    print(f"  Effect size (V)      : {effect_note}")

    # Logistic regression: only if Q53 is Likert (not applicable here)
    logit_note = (
        "Logistic regression not run: Q53 is not ordinal Likert-scale "
        "(detected as simple Yes/No/Maybe categorical)."
        if q53_type != "likert"
        else None
    )
    if logit_note:
        print(f"\n  {logit_note}")

    # ------------------------------------------------------------------
    # 6. Visuals + report
    # ------------------------------------------------------------------
    print("\n[6/6] Generating plot and report ...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_path = REPORTS_DIR / "q1_plot.png"
    _make_stacked_bar(work, q8_order, q53_order, plot_path)
    print(f"  Plot saved: {plot_path}")

    report_path = REPORTS_DIR / "q1_report.md"
    _write_report(
        path=report_path,
        col_q8=col_q8,
        col_q53=col_q53,
        q53_type=q53_type,
        n_before=n_before,
        n_final=n_final,
        work=work,
        q8_order=q8_order,
        q53_order=q53_order,
        ct_counts=ct_counts,
        ct_row_pct=ct_row_pct,
        ct_col_pct=ct_col_pct,
        chi2_stat=chi2_stat,
        dof=dof,
        p_val=p_val,
        v=v,
        sig_note=sig_note,
        effect_note=effect_note,
        logit_note=logit_note,
        plot_path=plot_path,
    )
    print(f"  Report saved: {report_path}")

    # Write a compact JSON summary for CI/PR comment consumption
    summary_path = REPORTS_DIR / "q1_summary.json"
    summary = {
        "n_final": int(n_final),
        "chi2": round(float(chi2_stat), 4),
        "dof": int(dof),
        "p_value": round(float(p_val), 4),
        "cramers_v": round(float(v), 4),
        "significance": sig_note,
        "effect_size": effect_note,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary JSON saved: {summary_path}")

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  N (after filtering)  : {n_final}")
    print(f"  Chi-square           : {chi2_stat:.4f}  (dof={dof})")
    print(f"  p-value              : {p_val:.4f}  → {sig_note}")
    print(f"  Cramér's V           : {v:.4f}  → {effect_note} effect")
    if logit_note:
        print(f"  Logistic regression  : skipped ({logit_note[:60]}...)")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _make_stacked_bar(
    work: pd.DataFrame,
    q8_order: list,
    q53_order: list,
    out_path: Path,
) -> None:
    """Stacked bar chart: Q53 distribution within each Q8 group (%)."""
    ct = pd.crosstab(work["Q8"], work["Q53"])
    # Re-index to maintain desired order
    ct = ct.reindex(index=q8_order, columns=q53_order, fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pct))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(q53_order)))  # type: ignore[attr-defined]

    for i, col in enumerate(q53_order):
        vals = pct[col].values if col in pct.columns else np.zeros(len(pct))
        bars = ax.bar(
            pct.index,
            vals,
            bottom=bottom,
            label=col,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add percentage labels inside bar segments ≥ 8 %
        for bar, v in zip(bars, vals):
            if v >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )
        bottom += vals

    ax.set_xlabel("Q8: Are you planning on obtaining your CPA license?", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title(
        "Distribution of 'MAcc Program Worth It' (Q53)\nby CPA License Intent (Q8)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(
        title="Q53: Was MAcc worth it?",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, fmt: str = "g") -> str:
    """Convert a DataFrame to a Markdown table string."""
    col_header = "| " + " | ".join([""] + list(map(str, df.columns))) + " |"
    sep = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows = []
    for idx, row in df.iterrows():
        if fmt == ".1f":
            vals = [f"{v:.1f}" for v in row]
        else:
            vals = [str(v) for v in row]
        rows.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join([col_header, sep] + rows)


def _write_report(
    path: Path,
    col_q8: str,
    col_q53: str,
    q53_type: str,
    n_before: int,
    n_final: int,
    work: pd.DataFrame,
    q8_order: list,
    q53_order: list,
    ct_counts: pd.DataFrame,
    ct_row_pct: pd.DataFrame,
    ct_col_pct: pd.DataFrame,
    chi2_stat: float,
    dof: int,
    p_val: float,
    v: float,
    sig_note: str,
    effect_note: str,
    logit_note,
    plot_path: Path,
) -> None:
    n_total = ct_counts.values.sum()

    q8_dist = work["Q8"].value_counts(sort=False)
    q53_dist = work["Q53"].value_counts(sort=False)

    q8_dist_lines = "\n".join(
        f"- **{k}**: {v} ({100*v/n_final:.1f}%)"
        for k, v in q8_dist.items()
    )
    q53_dist_lines = "\n".join(
        f"- **{k}**: {v} ({100*v/n_final:.1f}%)"
        for k, v in q53_dist.items()
    )

    report = textwrap.dedent(f"""\
    # Q1 Analysis Report
    ## CPA License Intent (Q8) vs. MAcc Program Worth It (Q53)

    **Research Question:**
    Is there a relationship between students planning to obtain their CPA license (Q8)
    and whether they felt the MAcc program was worth it (Q53)?

    ---

    ## 1. Data & Column Detection

    | Item | Value |
    | --- | --- |
    | Data file | `{DEFAULT_DATA_PATH}` |
    | Column used for Q8 | `{col_q8}` |
    | Column used for Q53 | `{col_q53}` |
    | Q8 question text | Are you planning on obtaining your CPA license? |
    | Q53 question text | Do you feel that the UVU MAcc program was worth the money you spent? |
    | Q53 response type detected | {q53_type} |

    ---

    ## 2. Cleaning Rules Applied

    1. **Header handling:** Qualtrics 3-row header format detected.
       Row 0 = question codes, rows 1–2 = question text and ImportId metadata (skipped).
    2. **Column resolution:** Exact match on `Q8` and `Q53`.
    3. **Missing values:** Rows where Q8 **or** Q53 is blank/null were dropped.
       Dropped {n_before - n_final} of {n_before} rows → **{n_final} usable responses**.
    4. **Q8 normalization:** Raw values trimmed and mapped:
       - `Yes` → `Yes / Plan to get CPA`
       - `No` → `No / Do not plan`
       - `Maybe` → `Unsure`
    5. **Q53 normalization:** Response type = `{q53_type}`. Values trimmed and kept as-is
       (Yes / No / Maybe). Not a Likert scale.

    ---

    ## 3. Sample Size

    | Stage | N |
    | --- | --- |
    | Raw rows loaded | {n_before} |
    | After dropping missing | {n_final} |

    ---

    ## 4. Category Distributions

    ### Q8 – CPA License Intent

    {q8_dist_lines}

    ### Q53 – MAcc Program Worth It

    {q53_dist_lines}

    ---

    ## 5. Contingency Table (Counts)

    {_md_table(ct_counts)}

    ---

    ## 6. Row-wise Percentages (within each Q8 group)

    {_md_table(ct_row_pct, fmt=".1f")}

    ---

    ## 7. Column-wise Percentages (within each Q53 group)

    {_md_table(ct_col_pct, fmt=".1f")}

    ---

    ## 8. Statistical Results

    | Statistic | Value |
    | --- | --- |
    | Chi-square (χ²) | {chi2_stat:.4f} |
    | Degrees of freedom | {dof} |
    | p-value | {p_val:.4f} |
    | Cramér's V | {v:.4f} |
    | Significance | {sig_note} |
    | Effect size | {effect_note} |

    ### Interpretation

    {"A statistically significant association" if p_val < 0.05 else "No statistically significant association"}
    was found between CPA license intent (Q8) and perceived program value (Q53)
    (χ²({dof}) = {chi2_stat:.2f}, p = {p_val:.3f}, Cramér's V = {v:.2f}).
    The effect size is **{effect_note}**.

    ---

    ## 9. Logistic Regression

    {logit_note if logit_note else "Logistic regression was run (see below)."}

    ---

    ## 10. Visualization

    ![Q1 Stacked Bar Chart](q1_plot.png)

    *Stacked bar chart showing the distribution of Q53 (Program Worth It) responses
    within each Q8 (CPA Intent) group.*

    ---

    *Report generated automatically by `scripts/q1_cpa_intent_vs_program_value.py`.*
    """)

    path.write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path_str = os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH))
    run_analysis(Path(data_path_str))
