"""
Quick descriptives for the synthetic Study 1 agenda item dataset.

This script is deliberately simple:
- Reads data/synthetic/study1_agenda_items_synthetic.csv
- Computes mean junior talk share and critical turns by condition
  and accountability flag
"""

import pandas as pd

def main():
    df = pd.read_csv("data/synthetic/study1_agenda_items_synthetic.csv")

    print("Raw rows:")
    print(df.head(), "\n")

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

if __name__ == "__main__":
    main()
