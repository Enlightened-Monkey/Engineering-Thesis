"""Plot the default Robbins–Monro schedules used in QH policy evaluation.

The script saves a PNG with three curves: eta_n, theta_n, and their ratio.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.qh_policy_evaluation import QHPolicyEvaluation

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")


def build_schedule(initial: float, exponent: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    schedule = QHPolicyEvaluation.robbins_monro_schedule(initial, exponent=exponent)
    steps = np.arange(n_steps)
    values = np.array([schedule(int(t)) for t in steps], dtype=float)
    return steps, values


def main() -> None:
    n_steps = 100000000

    # Defaults from QHPolicyEvaluation.
    eta_initial, eta_exponent = 0.2, 0.6
    theta_initial, theta_exponent = 0.05, 0.9

    steps, eta_values = build_schedule(eta_initial, eta_exponent, n_steps)
    _, theta_values = build_schedule(theta_initial, theta_exponent, n_steps)
    ratio = theta_values / eta_values

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "robbins_monro_schedules.png"

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(steps, eta_values, label="eta_n", color="tab:blue")
    axes[0].set_ylabel("eta_n")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(steps, theta_values, label="theta_n", color="tab:orange")
    axes[1].set_ylabel("theta_n")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(steps, ratio, label="eta_n / theta_n", color="tab:green")
    axes[2].set_ylabel("eta_n / theta_n")
    axes[2].set_xlabel("Iteration n")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Default Robbins–Monro Schedules (QH Policy Evaluation)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig.savefig(output_file, dpi=150)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
