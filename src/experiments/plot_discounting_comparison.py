#!/usr/bin/env python3
"""Plot a visual comparison between standard and quasi-hyperbolic discounting.

The script now produces a single, information-dense figure that contrasts
standard exponential discounting with quasi-hyperbolic (present-biased)
discounting. It overlays the discount-weight trajectories for different
present-bias (α) parameters and highlights how quasi-hyperbolic preferences
place comparatively more emphasis on early rewards.

Run the script directly to display the figure or save it to disk:
    python plot_discounting_comparison.py --output ../../data/plots/discounting_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def exponential_weights(gamma: float, horizon: int) -> np.ndarray:
    """Return exponential discount weights γ^t for t in [0, horizon)."""
    return gamma ** np.arange(horizon)


def quasi_hyperbolic_weights(alpha: float, gamma: float, horizon: int) -> np.ndarray:
    """Return quasi-hyperbolic discount weights with present-bias α."""
    weights = alpha * (gamma ** np.arange(horizon))
    weights[0] = 1.0  # immediate reward receives full weight
    return weights


def parse_alpha_values(raw_values: str) -> List[float]:
    """Parse comma-separated α values from CLI."""
    return [float(value.strip()) for value in raw_values.split(',') if value.strip()]


def ensure_output_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_discounting(
    gamma: float,
    alpha_values: Iterable[float],
    horizon: int,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Path | None:
    """Create the comparison plot and optionally save it to disk."""
    alpha_values = sorted(set(float(s) for s in alpha_values))

    if len(alpha_values) == 0:
        raise ValueError("At least one α value must be provided.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    time_steps = np.arange(horizon)
    exp_weights = exponential_weights(gamma, horizon)

    ax.plot(
        time_steps,
        exp_weights,
        label="Exponential (α=1.0)",
        linewidth=3,
        color="#1b9e77",
    )

    qh_weights_by_alpha = {}
    for alpha in alpha_values:
        weights = quasi_hyperbolic_weights(alpha, gamma, horizon)
        qh_weights_by_alpha[alpha] = weights
        ax.plot(
            time_steps,
            weights,
            label=f"Quasi-hyperbolic (α={alpha:.2f})",
            linewidth=2,
        )

    ax.set_title("Discount weights by time step", fontsize=14)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Discount weight")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

 

    early_window = min(5, horizon - 1)
    ax.axvspan(
        time_steps[0],
        early_window,
        color="#e0f3f8",
        alpha=0.5,
        label="Wczesny horyzont",
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    fig.suptitle(
        "Standard vs quasi-hiperboliczne dyskontowanie nagród",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = output_path.with_suffix(".png") if output_path.suffix == "" else output_path
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot comparisons between standard and quasi-hyperbolic discounting",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor γ for both models (default: 0.95)",
    )
    parser.add_argument(
        "--alpha-values",
        type=parse_alpha_values,
        default=[0.6, 0.8, 0.95],
        help=(
            "Comma-separated α values for quasi-hyperbolic discounting. "
            "Example: 0.4,0.7,0.95"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Number of time steps to display (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure (PNG). Directories are created as needed.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; useful for headless environments.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not 0 < args.gamma < 1:
        raise ValueError("γ must be in the open interval (0, 1).")

    if any(not 0 <= alpha <= 1 for alpha in args.alpha_values):
        raise ValueError("All α values must be in the interval [0, 1].")

    if args.horizon < 5:
        raise ValueError("Time horizon should be at least 5 to allow informative plots.")

    output_path = ensure_output_path(args.output)
    saved_path = plot_discounting(
        gamma=args.gamma,
        alpha_values=args.alpha_values,
        horizon=args.horizon,
        output_path=output_path,
        show=not args.no_show,
    )

    if saved_path is not None:
        print(f"Figure saved to: {saved_path}")


if __name__ == "__main__":
    main()
