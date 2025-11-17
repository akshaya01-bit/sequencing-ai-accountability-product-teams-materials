"""
Generate a richer synthetic agenda-item-level dataset for Study 1.

This script simulates:
- Multiple teams
- Multiple meetings per team
- Multiple agenda items per meeting
- Three sequencing conditions: AI_FIRST, HUMAN_FIRST, STATUS_QUO
- Accountability flag
- Outcomes: junior_talk_share, junior_critical_turns, override_ai, psych_safety_score

The goal is to mirror the structure of the planned analysis while using
non-sensitive synthetic data.
"""

import csv
import random

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def main():
    random.seed(42)

    teams = [f"T{i}" for i in range(1, 7)]  # T1..T6
    n_meetings_per_team = 4
    n_items_per_meeting = 5

    conditions = ["AI_FIRST", "HUMAN_FIRST", "STATUS_QUO"]

    rows = []

    for t in teams:
        for m in range(1, n_meetings_per_team + 1):
            meeting_id = f"{t}_M{m}"
            week = m  # simple: meeting index as week

            for a in range(1, n_items_per_meeting + 1):
                agenda_item_id = f"{meeting_id}_A{a}"

                seq_cond = random.choice(conditions)
                accountability = random.choice([0, 1])

                # Base junior talk share around 0.22
                base = random.normalvariate(0.22, 0.05)

                # Effects depending on condition/accountability
                delta = 0.0
                if seq_cond == "HUMAN_FIRST":
                    delta += 0.04
                if seq_cond == "AI_FIRST":
                    delta -= 0.01
                if accountability == 1:
                    delta += 0.03
                # interaction bump for HUMAN_FIRST + accountability
                if seq_cond == "HUMAN_FIRST" and accountability == 1:
                    delta += 0.03

                junior_talk_share = clamp(base + delta, 0.05, 0.75)

                # junior critical turns roughly aligned with talk share
                expected_crit = 2.0 * junior_talk_share * 5  # 0..7-ish
                # simple integer with some noise
                junior_critical_turns = max(0, int(round(random.normalvariate(expected_crit, 1.0))))

                # override_ai more likely when juniors speak more + accountability
                override_prob = 0.10
                if seq_cond == "HUMAN_FIRST":
                    override_prob += 0.08
                if accountability == 1:
                    override_prob += 0.07
                if junior_talk_share > 0.3:
                    override_prob += 0.05
                override_prob = clamp(override_prob, 0.02, 0.6)
                override_ai = 1 if random.random() < override_prob else 0

                # psych safety roughly aligned with accountability + junior talk
                psych_safety = 3.4
                if accountability == 1:
                    psych_safety += 0.3
                if junior_talk_share > 0.3:
                    psych_safety += 0.2
                psych_safety += random.normalvariate(0.0, 0.2)
                psych_safety = clamp(psych_safety, 2.0, 5.0)

                rows.append(
                    dict(
                        team_id=t,
                        meeting_id=meeting_id,
                        week=week,
                        agenda_item_id=agenda_item_id,
                        sequence_condition=seq_cond,
                        accountability=accountability,
                        junior_talk_share=round(junior_talk_share, 3),
                        junior_critical_turns=junior_critical_turns,
                        override_ai=override_ai,
                        psych_safety_score=round(psych_safety, 2),
                    )
                )

    out_path = "data/synthetic/study1_agenda_items_synthetic_full.csv"
    fieldnames = [
        "team_id",
        "meeting_id",
        "week",
        "agenda_item_id",
        "sequence_condition",
        "accountability",
        "junior_talk_share",
        "junior_critical_turns",
        "override_ai",
        "psych_safety_score",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} synthetic rows to {out_path}")

if __name__ == "__main__":
    main()
