"""
Quick descriptives for the synthetic Study 1 agenda item dataset.

This script:
- Reads data/synthetic/study1_agenda_items_synthetic.csv
- Computes mean junior talk share, critical turns, override rate, and
  psych safety by sequence_condition x accountability
- Prints the table
- Saves it to fig/descriptives_by_condition_synthetic.csv
"""

import os
import pandas as pd

def main():
    os.makedirs("fig", exist_ok=True)

    df = pd.read_csv("data/synthetic/study1_agenda_items_synthetic.csv")

    grouped = (
        df
        .groupby(["sequence_condition", "accountability"])
        .agg(
            mean_junior_talk_share=("junior_talk_share", "mean"),
            mean_junior_critical_turns=("junior_critical_turns", "mean"),
            override_rate=("override_ai", "mean"),
            mean_psych_safety=("psych_safety_score", "mean"),
        )
        .reset_index()
    )

    print("Descriptives by condition x accountability:")
    print(grouped.to_string(index=False))

    out_path = "fig/descriptives_by_condition_synthetic.csv"
    grouped.to_csv(out_path, index=False)
    print(f"\nSaved descriptives table to {out_path}")

if __name__ == "__main__":
    main()
