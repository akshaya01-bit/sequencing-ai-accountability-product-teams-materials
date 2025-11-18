"""
Orchestrator script to run the full synthetic pipeline:

1. Generate agenda-level synthetic data (if scripts exist)
2. Run descriptives and main regression
3. Run power simulation
4. Generate turn-level synthetic data
5. Train the critical-turn classifier

Outputs are written under data/synthetic, fig/, and models/.

Log:
- fig/run_all_log.txt
"""

import os
import subprocess
import sys
from datetime import datetime


def run(cmd, log_f):
    log_f.write(f"\n[{datetime.utcnow().isoformat()}] Running: {cmd}\n")
    log_f.flush()
    try:
        subprocess.run(cmd, shell=True, check=True)
        log_f.write("  -> OK\n")
    except subprocess.CalledProcessError as e:
        log_f.write(f"  -> FAILED with code {e.returncode}\n")


def main():
    os.makedirs("fig", exist_ok=True)

    with open("fig/run_all_log.txt", "w") as log_f:
        log_f.write("Run-all synthetic pipeline log\n")
        log_f.write("================================\n")
        log_f.write(f"Started at (UTC): {datetime.utcnow().isoformat()}\n")

        # 1. Generate agenda-level synthetic data
        if os.path.exists("code/analysis/generate_synthetic_agenda_data.py"):
            run("python3 code/analysis/generate_synthetic_agenda_data.py", log_f)

        # 2. Descriptives and main regression
        if os.path.exists("code/analysis/quick_descriptives.py"):
            run("python3 code/analysis/quick_descriptives.py", log_f)

        if os.path.exists("code/analysis/main_regression_synthetic.py"):
            run("python3 code/analysis/main_regression_synthetic.py", log_f)

        # 3. Power simulation
        if os.path.exists("code/analysis/power_simulation_study1.py"):
            run("python3 code/analysis/power_simulation_study1.py", log_f)

        # 4. Turn-level synthetic data
        if os.path.exists("code/ml/generate_synthetic_turns.py"):
            run("python3 code/ml/generate_synthetic_turns.py", log_f)

        # 5. Critical-turn classifier training
        if os.path.exists("code/ml/train_critical_turn_classifier.py"):
            run("python3 code/ml/train_critical_turn_classifier.py", log_f)

        log_f.write(f"\nFinished at (UTC): {datetime.utcnow().isoformat()}\n")


if __name__ == "__main__":
    main()
