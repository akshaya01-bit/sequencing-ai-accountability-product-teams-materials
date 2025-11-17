"""
Main regression analysis on the richer synthetic agenda-item dataset.

Implements a model of the form:

  junior_talk_share ~ AI_first + Human_first + accountability
                      + AI_first:accountability
                      + Human_first:accountability
                      + team fixed effects

using statsmodels with robust (HC1) standard errors.

This script:
- Reads data/synthetic/study1_agenda_items_synthetic_full.csv
- Fits the model
- Prints results to the console
- Saves:
  - fig/regression_results_synthetic.csv (coefficients table)
  - fig/regression_summary_synthetic.txt (full text summary)
"""

import os
import pandas as pd
import statsmodels.formula.api as smf

def main():
    # Ensure fig directory exists
    os.makedirs("fig", exist_ok=True)

    df = pd.read_csv("data/synthetic/study1_agenda_items_synthetic_full.csv")

    # Create dummy variables for sequencing conditions
    df["AI_first"] = (df["sequence_condition"] == "AI_FIRST").astype(int)
    df["Human_first"] = (df["sequence_condition"] == "HUMAN_FIRST").astype(int)
    # STATUS_QUO is the omitted baseline

    # Build the formula: note the interactions and team fixed effects
    formula = (
        "junior_talk_share ~ AI_first + Human_first + accountability "
        "+ AI_first:accountability + Human_first:accountability "
        "+ C(team_id)"
    )

    model = smf.ols(formula=formula, data=df)
    results = model.fit(cov_type="HC1")  # robust SEs

    print("=== OLS results with robust (HC1) SEs ===")
    print(results.summary())

    # Save full text summary
    summary_path = "fig/regression_summary_synthetic.txt"
    with open(summary_path, "w") as f:
        f.write(results.summary().as_text())
    print(f"\nSaved full summary to {summary_path}")

    # Save a compact coefficient table (estimates, SE, p-values, etc.)
    coefs = results.summary2().tables[1]
    coef_path = "fig/regression_results_synthetic.csv"
    coefs.to_csv(coef_path)
    print(f"Saved coefficient table to {coef_path}")

if __name__ == "__main__":
    main()
