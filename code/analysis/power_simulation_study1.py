"""
Simulation-based power analysis for Study 1.

We simulate the same general structure as the main synthetic dataset:
- Teams
- Meetings per team
- Agenda items per meeting
- Sequencing conditions: AI_FIRST / HUMAN_FIRST / STATUS_QUO
- Accountability flag

For each simulated dataset, we fit the same regression model as in
main_regression_synthetic.py and record whether key coefficients are
significant at alpha = 0.05.

Outputs:
- data/synthetic/power_simulation_results_study1.csv
- fig/power_curve_study1.csv
- fig/power_simulation_summary_study1.txt
"""

import csv
import os
import random

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def simulate_dataset(
    n_teams: int,
    n_meetings_per_team: int,
    n_items_per_meeting: int,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    teams = [f"T{i}" for i in range(1, n_teams + 1)]
    conditions = ["AI_FIRST", "HUMAN_FIRST", "STATUS_QUO"]

    rows = []

    for t in teams:
        for m in range(1, n_meetings_per_team + 1):
            meeting_id = f"{t}_M{m}"
            week = m

            for a in range(1, n_items_per_meeting + 1):
                agenda_item_id = f"{meeting_id}_A{a}"

                seq_cond = random.choice(conditions)
                accountability = random.choice([0, 1])

                # Base junior talk share around 0.22
                base = np.random.normal(0.22, 0.05)

                # True effect sizes (assumptions for power)
                delta = 0.0
                if seq_cond == "HUMAN_FIRST":
                    delta += 0.05        # main effect of HUMAN_FIRST
                if seq_cond == "AI_FIRST":
                    delta -= 0.02        # slight negative main effect
                if accountability == 1:
                    delta += 0.04        # main effect of accountability
                if seq_cond == "HUMAN_FIRST" and accountability == 1:
                    delta += 0.05        # interaction (our key effect)

                junior_talk_share = clamp(base + delta, 0.05, 0.80)

                rows.append(
                    dict(
                        team_id=t,
                        meeting_id=meeting_id,
                        week=week,
                        agenda_item_id=agenda_item_id,
                        sequence_condition=seq_cond,
                        accountability=accountability,
                        junior_talk_share=junior_talk_share,
                    )
                )

    df = pd.DataFrame(rows)
    df["AI_first"] = (df["sequence_condition"] == "AI_FIRST").astype(int)
    df["Human_first"] = (df["sequence_condition"] == "HUMAN_FIRST").astype(int)
    return df


def run_power_simulation(
    n_sims: int = 300,
    n_teams: int = 6,
    n_meetings_per_team: int = 4,
    n_items_per_meeting: int = 5,
    alpha: float = 0.05,
):
    os.makedirs("data/synthetic", exist_ok=True)
    os.makedirs("fig", exist_ok=True)

    results = []

    for sim_id in range(1, n_sims + 1):
        df = simulate_dataset(
            n_teams=n_teams,
            n_meetings_per_team=n_meetings_per_team,
            n_items_per_meeting=n_items_per_meeting,
            seed=sim_id,
        )

        formula = (
            "junior_talk_share ~ AI_first + Human_first + accountability "
            "+ AI_first:accountability + Human_first:accountability "
            "+ C(team_id)"
        )

        try:
            model = smf.ols(formula=formula, data=df)
            fit = model.fit(cov_type="HC1")
            coefs = fit.summary2().tables[1]

            def is_sig(term):
                if term not in coefs.index:
                    return np.nan
                pval = coefs.loc[term, "P>|t|"]
                return 1 if pval < alpha else 0

            sig_AI_first = is_sig("AI_first")
            sig_Human_first = is_sig("Human_first")
            sig_HF_Acc = is_sig("Human_first:accountability")

            results.append(
                dict(
                    sim_id=sim_id,
                    n_teams=n_teams,
                    n_meetings_per_team=n_meetings_per_team,
                    n_items_per_meeting=n_items_per_meeting,
                    alpha=alpha,
                    sig_AI_first=sig_AI_first,
                    sig_Human_first=sig_Human_first,
                    sig_HumanFirst_Acc=sig_HF_Acc,
                    converged=1,
                )
            )
        except Exception as e:
            results.append(
                dict(
                    sim_id=sim_id,
                    n_teams=n_teams,
                    n_meetings_per_team=n_meetings_per_team,
                    n_items_per_meeting=n_items_per_meeting,
                    alpha=alpha,
                    sig_AI_first=np.nan,
                    sig_Human_first=np.nan,
                    sig_HumanFirst_Acc=np.nan,
                    converged=0,
                )
            )

    df_res = pd.DataFrame(results)
    out_path = "data/synthetic/power_simulation_results_study1.csv"
    df_res.to_csv(out_path, index=False)

    # Aggregate power (proportion significant among converged sims)
    df_conv = df_res[df_res["converged"] == 1]
    summary = {
        "n_sims": len(df_res),
        "n_converged": len(df_conv),
        "power_AI_first": df_conv["sig_AI_first"].mean(),
        "power_Human_first": df_conv["sig_Human_first"].mean(),
        "power_HumanFirst_Acc": df_conv["sig_HumanFirst_Acc"].mean(),
        "alpha": alpha,
        "n_teams": n_teams,
        "n_meetings_per_team": n_meetings_per_team,
        "n_items_per_meeting": n_items_per_meeting,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("fig/power_curve_study1.csv", index=False)

    # Human-readable summary
    lines = [
        "Power simulation summary for Study 1",
        "====================================",
        f"n_sims: {summary['n_sims']}",
        f"n_converged: {summary['n_converged']}",
        "",
        f"alpha: {alpha}",
        f"n_teams: {n_teams}",
        f"n_meetings_per_team: {n_meetings_per_team}",
        f"n_items_per_meeting: {n_items_per_meeting}",
        "",
        f"Estimated power (AI_first main effect): "
        f"{summary['power_AI_first']:.3f}",
        f"Estimated power (Human_first main effect): "
        f"{summary['power_Human_first']:.3f}",
        f"Estimated power (Human_first x accountability interaction): "
        f"{summary['power_HumanFirst_Acc']:.3f}",
    ]
    with open("fig/power_simulation_summary_study1.txt", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved detailed results to {out_path}")
    print("Saved aggregate power curve to fig/power_curve_study1.csv")
    print("Saved text summary to fig/power_simulation_summary_study1.txt")


if __name__ == "__main__":
    run_power_simulation()
