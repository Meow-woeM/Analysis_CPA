    # Q1 Analysis Report
    ## CPA License Intent (Q8) vs. MAcc Program Worth It (Q53)

    **Research Question:**
    Is there a relationship between students planning to obtain their CPA license (Q8)
    and whether they felt the MAcc program was worth it (Q53)?

    ---

    ## 1. Data & Column Detection

    | Item | Value |
    | --- | --- |
    | Data file | `data/Grad Program Exit Survey Data 2024.xlsx` |
    | Column used for Q8 | `Q8` |
    | Column used for Q53 | `Q53` |
    | Q8 question text | Are you planning on obtaining your CPA license? |
    | Q53 question text | Do you feel that the UVU MAcc program was worth the money you spent? |
    | Q53 response type detected | simple |

    ---

    ## 2. Cleaning Rules Applied

    1. **Header handling:** Qualtrics 3-row header format detected.
       Row 0 = question codes, rows 1–2 = question text and ImportId metadata (skipped).
    2. **Column resolution:** Exact match on `Q8` and `Q53`.
    3. **Missing values:** Rows where Q8 **or** Q53 is blank/null were dropped.
       Dropped 10 of 65 rows → **55 usable responses**.
    4. **Q8 normalization:** Raw values trimmed and mapped:
       - `Yes` → `Yes / Plan to get CPA`
       - `No` → `No / Do not plan`
       - `Maybe` → `Unsure`
    5. **Q53 normalization:** Response type = `simple`. Values trimmed and kept as-is
       (Yes / No / Maybe). Not a Likert scale.

    ---

    ## 3. Sample Size

    | Stage | N |
    | --- | --- |
    | Raw rows loaded | 65 |
    | After dropping missing | 55 |

    ---

    ## 4. Category Distributions

    ### Q8 – CPA License Intent

    - **Yes / Plan to get CPA**: 50 (90.9%)
- **Unsure**: 4 (7.3%)
- **No / Do not plan**: 1 (1.8%)

    ### Q53 – MAcc Program Worth It

    - **Yes**: 39 (70.9%)
- **Maybe**: 12 (21.8%)
- **No**: 4 (7.3%)

    ---

    ## 5. Contingency Table (Counts)

    |  | Yes | Maybe | No |
| --- | --- | --- | --- |
| Yes / Plan to get CPA | 37 | 11 | 2 |
| Unsure | 2 | 1 | 1 |
| No / Do not plan | 0 | 0 | 1 |

    ---

    ## 6. Row-wise Percentages (within each Q8 group)

    |  | Yes | Maybe | No |
| --- | --- | --- | --- |
| Yes / Plan to get CPA | 74.0 | 22.0 | 4.0 |
| Unsure | 50.0 | 25.0 | 25.0 |
| No / Do not plan | 0.0 | 0.0 | 100.0 |

    ---

    ## 7. Column-wise Percentages (within each Q53 group)

    |  | Yes | Maybe | No |
| --- | --- | --- | --- |
| Yes / Plan to get CPA | 94.9 | 91.7 | 50.0 |
| Unsure | 5.1 | 8.3 | 25.0 |
| No / Do not plan | 0.0 | 0.0 | 25.0 |

    ---

    ## 8. Statistical Results

    | Statistic | Value |
    | --- | --- |
    | Chi-square (χ²) | 15.5481 |
    | Degrees of freedom | 4 |
    | p-value | 0.0037 |
    | Cramér's V | 0.3760 |
    | Significance | p < 0.01 (significant) |
    | Effect size | medium |

    ### Interpretation

    A statistically significant association
    was found between CPA license intent (Q8) and perceived program value (Q53)
    (χ²(4) = 15.55, p = 0.004, Cramér's V = 0.38).
    The effect size is **medium**.

    ---

    ## 9. Logistic Regression

    Logistic regression not run: Q53 is not ordinal Likert-scale (detected as simple Yes/No/Maybe categorical).

    ---

    ## 10. Visualization

    ![Q1 Stacked Bar Chart](q1_plot.png)

    *Stacked bar chart showing the distribution of Q53 (Program Worth It) responses
    within each Q8 (CPA Intent) group.*

    ---

    *Report generated automatically by `scripts/q1_cpa_intent_vs_program_value.py`.*
