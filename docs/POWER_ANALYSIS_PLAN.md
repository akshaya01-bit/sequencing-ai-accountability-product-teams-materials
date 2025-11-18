# Power analysis plan for Study 1

This document describes the simulation-based power analysis implemented in:

- `code/analysis/power_simulation_study1.py`

for the study:

> **Sequencing AI Recommendations and Accountability Practices in Product Teams**

---

## 1. Design assumptions

The simulation mirrors a stylized version of the planned field experiment:

- Units:
  - Product **teams** working with an AI copilot.
  - Repeated **meetings** per team.
  - Multiple **agenda items** per meeting.

- Factors:
  - `sequence_condition` ∈ {`AI_FIRST`, `HUMAN_FIRST`, `STATUS_QUO`}
  - `accountability` ∈ {0,1}

- Outcome:
  - `junior_talk_share` – proportion of speaking time accounted for by junior
    contributors during each agenda item.

We assume a fixed number of:

- `n_teams` (e.g., 6 in the default script)
- `n_meetings_per_team`
- `n_items_per_meeting`

These can be modified directly in `run_power_simulation()`.

---

## 2. Data-generating process

For each simulated dataset:

1. We create a grid of teams × meetings × agenda items.
2. For each agenda item, we assign:
   - A sequencing condition (`AI_FIRST`, `HUMAN_FIRST`, `STATUS_QUO`), chosen at random.
   - An accountability flag (0/1), also randomized.

3. We construct a synthetic `junior_talk_share` using:

   - A baseline value near 0.22.
   - Additive effects for:
     - `HUMAN_FIRST` (positive main effect)
     - `AI_FIRST` (slight negative main effect)
     - `accountability` (positive main effect)
     - `HUMAN_FIRST × accountability` (positive interaction; the key effect of interest).

These effect sizes are not empirical estimates; they encode the *direction* of
hypothesized effects and are used solely for power calculations.

---

## 3. Model specification

For each simulated dataset, we fit the same OLS model as in the main synthetic
analysis:

```text
junior_talk_share ~ AI_first + Human_first + accountability
                    + AI_first:accountability
                    + Human_first:accountability
                    + C(team_id)
where:

AI_first and Human_first are dummy variables for the sequencing conditions
(STATUS_QUO is the omitted baseline).

accountability is a 0/1 indicator.

Interaction terms capture whether accountability amplifies/dampens the effect
of each sequencing condition.

C(team_id) adds team fixed effects.

We use robust (HC1) standard errors.

4. Power estimation

The script runs n_sims independent simulations (default: 300). For each:

Generate a synthetic dataset.

Fit the model above.

Record whether key coefficients are statistically significant at α = 0.05:

AI_first main effect.

Human_first main effect.

Human_first:accountability interaction.

The empirical power for each coefficient is estimated as:

power = (# of simulations where p < alpha) / (# of converged simulations)


These results are written to:

data/synthetic/power_simulation_results_study1.csv
(one row per simulation, including convergence flags and significance indicators).

fig/power_curve_study1.csv
(summary row with estimated power for each effect and the design parameters).

fig/power_simulation_summary_study1.txt
(human-readable summary of the power estimates).

5. Interpretation

The power simulation answers questions such as:

Given the assumed effect sizes and design (number of teams, meetings, and
agenda items), what is the probability of detecting:

A positive HUMAN_FIRST main effect?

A positive HUMAN_FIRST × accountability interaction?

If estimated power for a key effect (especially the interaction) is
substantially below, say, 0.8, the design can be adjusted by:

Increasing the number of teams.

Increasing the number of meetings per team.

Increasing the number of agenda items per meeting.

These adjustments can be explored by editing the arguments to
run_power_simulation() in power_simulation_study1.py and re-running.

6. Limitations

This simulation:

Uses a simplified data-generating process and does not capture all possible
sources of heterogeneity (e.g., different variances across teams).

Focuses on junior_talk_share as the primary outcome; secondary outcomes
(e.g., overrides, critical turns) are not powered here.

Nonetheless, it provides a transparent and reproducible approximation for
planning the study’s sample size and structure.


