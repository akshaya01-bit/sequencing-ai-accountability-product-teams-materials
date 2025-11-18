"""
Generate synthetic turn-level data for Study 1.

Each row is a single utterance ("turn") with:
- speaker_role
- text
- whether the speaker is junior
- whether the turn is labeled as "critical"
- experimental conditions (sequence_condition, accountability, ai_suggestion_visible)

Output:
- data/synthetic/study1_turns_labeled_synthetic.csv
"""

import csv
import os
import random


def make_text(is_critical: int, sequence_condition: str, accountability: int) -> str:
    base = "Synthetic utterance"
    if is_critical:
        mood = "raising a concern about trade-offs and risks"
    else:
        mood = "offering neutral progress updates"

    cond = f"under {sequence_condition} with "
    cond += "accountability" if accountability else "no accountability"

    return f"{base} {mood} {cond}."


def main():
    random.seed(123)
    os.makedirs("data/synthetic", exist_ok=True)

    teams = [f"T{i}" for i in range(1, 7)]
    sequence_conditions = ["AI_FIRST", "HUMAN_FIRST", "STATUS_QUO"]

    rows = []
    turn_id_counter = 1

    for t in teams:
        for m in range(1, 5):  # meetings per team
            meeting_id = f"{t}_M{m}"
            accountability = random.choice([0, 1])
            seq_cond = random.choice(sequence_conditions)

            for a in range(1, 4):  # agenda items per meeting (simpler here)
                agenda_item_id = f"{meeting_id}_A{a}"

                # Simulate 6 turns per agenda item
                for pos in range(1, 7):
                    speaker_role = random.choice(
                        ["junior_eng", "junior_pm", "senior_eng", "director"]
                    )
                    is_junior = 1 if speaker_role.startswith("junior") else 0

                    # Base probability of critical turn
                    p_crit = 0.05
                    if is_junior:
                        p_crit += 0.05
                    if seq_cond == "HUMAN_FIRST":
                        p_crit += 0.05
                    if accountability == 1:
                        p_crit += 0.05
                    if is_junior and seq_cond == "HUMAN_FIRST" and accountability == 1:
                        p_crit += 0.10

                    p_crit = max(0.02, min(0.6, p_crit))
                    is_critical = 1 if random.random() < p_crit else 0

                    ai_suggestion_visible = 1 if seq_cond == "AI_FIRST" else 0

                    text = make_text(is_critical, seq_cond, accountability)

                    rows.append(
                        dict(
                            team_id=t,
                            meeting_id=meeting_id,
                            agenda_item_id=agenda_item_id,
                            turn_id=turn_id_counter,
                            turn_position=pos,
                            speaker_role=speaker_role,
                            is_junior=is_junior,
                            text=text,
                            is_critical=is_critical,
                            ai_suggestion_visible=ai_suggestion_visible,
                            sequence_condition=seq_cond,
                            accountability=accountability,
                        )
                    )
                    turn_id_counter += 1

    out_path = "data/synthetic/study1_turns_labeled_synthetic.csv"
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} synthetic turns to {out_path}")


if __name__ == "__main__":
    main()
