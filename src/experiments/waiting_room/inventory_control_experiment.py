"""Train and evaluate QH Q-learning on the finite inventory control problem."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - allow execution via `python path/to/script.py`
    from ..algorithms.qh_qlearning import QHQLearning
    from ..models.mdp_environments import InventoryControlMDP
except ImportError:  # pragma: no cover
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    for candidate in (REPO_ROOT, SRC_ROOT):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
    from algorithms.qh_qlearning import QHQLearning  # type: ignore
    from models.mdp_environments import InventoryControlMDP  # type: ignore


@dataclass
class InventoryEnvironmentConfig:
    """Environment parameters mirroring the inventory control example."""

    max_inventory: int = 2
    procurement_cost: float = 5.0
    holding_cost: float = 2.0
    selling_price: float = 9.0
    demand_support: Sequence[int] = (0, 1, 2)
    demand_prob: Sequence[float] = (0.2, 0.3, 0.5)
    initial_state: int = 0


@dataclass
class TrainingConfig:
    """Hyper-parameters for training and evaluation."""

    episodes: int = 5_000
    episode_length: int = 30
    sigma: float = 0.3
    gamma: float = 0.9
    alpha: float = 0.1
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    eval_episodes: int = 250
    seed: Optional[int] = None
    results_dir: Path = Path("data/results")
    save_summary: bool = True
    environment: InventoryEnvironmentConfig = field(default_factory=InventoryEnvironmentConfig)


def parse_args() -> TrainingConfig:
    env_defaults = InventoryEnvironmentConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Train quasi-hyperbolic Q-learning on the finite inventory control "
            "benchmark with demand following the distribution (0.2, 0.3, 0.5)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes, help="Training episodes")
    parser.add_argument("--episode-length", type=int, default=TrainingConfig.episode_length, help="Steps per episode (finite-horizon rollout)")
    parser.add_argument("--sigma", type=float, default=TrainingConfig.sigma, help="Present-bias parameter σ")
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma, help="Exponential discount factor γ")
    parser.add_argument("--alpha", type=float, default=TrainingConfig.alpha, help="Learning rate α")
    parser.add_argument("--epsilon", type=float, default=TrainingConfig.epsilon, help="Initial ε for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=TrainingConfig.epsilon_decay, help="Multiplicative decay applied after each episode")
    parser.add_argument("--min-epsilon", type=float, default=TrainingConfig.min_epsilon, help="Floor for ε during decay")
    parser.add_argument("--eval-episodes", type=int, default=TrainingConfig.eval_episodes, help="Evaluation episodes after training")
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed, help="Random seed for reproducibility")
    parser.add_argument("--results-dir", type=Path, default=TrainingConfig.results_dir, help="Directory for saving JSON summaries")
    parser.add_argument("--no-save", action="store_true", help="Skip saving summary JSON to disk")

    parser.add_argument("--max-inventory", type=int, default=env_defaults.max_inventory, help="Inventory capacity M")
    parser.add_argument("--procurement-cost", type=float, default=env_defaults.procurement_cost, help="Procurement cost c per ordered unit")
    parser.add_argument("--holding-cost", type=float, default=env_defaults.holding_cost, help="Holding cost h per leftover unit")
    parser.add_argument("--selling-price", type=float, default=env_defaults.selling_price, help="Selling price p per unit sold")
    parser.add_argument("--initial-state", type=int, default=env_defaults.initial_state, help="Initial inventory level s₀")

    parser.add_argument(
        "--demand-support",
        type=str,
        default=",".join(str(x) for x in env_defaults.demand_support),
        help="Comma-separated integers for demand support",
    )
    parser.add_argument(
        "--demand-prob",
        type=str,
        default=",".join(str(x) for x in env_defaults.demand_prob),
        help="Comma-separated probabilities matching demand support",
    )

    args = parser.parse_args()

    demand_support = tuple(int(x.strip()) for x in args.demand_support.split(",") if x.strip())
    demand_prob = tuple(float(x.strip()) for x in args.demand_prob.split(",") if x.strip())

    if len(demand_support) != len(demand_prob):
        raise ValueError("Demand support and probabilities must have the same length.")

    env_config = InventoryEnvironmentConfig(
        max_inventory=args.max_inventory,
        procurement_cost=args.procurement_cost,
        holding_cost=args.holding_cost,
        selling_price=args.selling_price,
        demand_support=demand_support,
        demand_prob=demand_prob,
        initial_state=args.initial_state,
    )

    return TrainingConfig(
        episodes=args.episodes,
        episode_length=args.episode_length,
        sigma=args.sigma,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        results_dir=args.results_dir,
        save_summary=not args.no_save,
        environment=env_config,
    )


def build_env(cfg: InventoryEnvironmentConfig) -> InventoryControlMDP:
    return InventoryControlMDP(
        max_inventory=cfg.max_inventory,
        procurement_cost=cfg.procurement_cost,
        holding_cost=cfg.holding_cost,
        selling_price=cfg.selling_price,
        demand_support=np.asarray(cfg.demand_support, dtype=int),
        demand_prob=np.asarray(cfg.demand_prob, dtype=float),
        initial_state=cfg.initial_state,
    )


def decay_epsilon(agent: QHQLearning, config: TrainingConfig) -> None:
    agent.epsilon = max(config.min_epsilon, agent.epsilon * config.epsilon_decay)


def train_agent(config: TrainingConfig) -> Dict[str, object]:
    if config.seed is not None:
        np.random.seed(config.seed)

    env = build_env(config.environment)
    agent = QHQLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        sigma=config.sigma,
        gamma=config.gamma,
        alpha=config.alpha,
        epsilon=config.epsilon,
    )

    episode_rewards: List[float] = []
    start_time = time.time()

    for episode in range(config.episodes):
        state = env.reset()
        cumulative_reward = 0.0

        for _ in range(config.episode_length):
            action = agent.get_action(state, exploration=True)
            next_state, reward, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            cumulative_reward += float(reward)

        episode_rewards.append(cumulative_reward)
        decay_epsilon(agent, config)

        if (episode + 1) % max(1, config.episodes // 10) == 0:
            print(
                f"Episode {episode + 1}/{config.episodes}: "
                f"reward={cumulative_reward:.2f}, epsilon={agent.epsilon:.3f}"
            )

    training_time = time.time() - start_time

    evaluation = evaluate_agent(agent, config.environment, config.episode_length, config.eval_episodes)

    results = {
        "config": {
            "training": asdict(config),
            "environment": asdict(config.environment),
        },
        "training_stats": {
            "episodes": config.episodes,
            "episode_length": config.episode_length,
            "training_time_s": training_time,
            "episode_rewards": episode_rewards,
            "final_policy": agent.get_policy().tolist(),
            "final_values": agent.get_value_function().tolist(),
            "final_epsilon": agent.epsilon,
        },
        "evaluation": evaluation,
    }

    if config.save_summary:
        config.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = config.results_dir / f"inventory_control_qh_{timestamp}.json"
        with save_path.open("w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        print(f"Saved summary to {save_path}")

    print("\nEvaluation summary:")
    print(
        f"  Avg reward per episode: {evaluation['avg_reward']:.3f} ± {evaluation['std_reward']:.3f}"
    )
    print(f"  Avg order qty        : {evaluation['avg_order']:.3f}")
    print(f"  Avg sales            : {evaluation['avg_sales']:.3f}")
    print(f"  Avg ending inventory : {evaluation['avg_ending_inventory']:.3f}")

    return results


def evaluate_agent(
    agent: QHQLearning,
    env_config: InventoryEnvironmentConfig,
    horizon: int,
    episodes: int,
) -> Dict[str, float]:
    env = build_env(env_config)

    rewards: List[float] = []
    orders: List[float] = []
    sales: List[float] = []
    ending_inventory: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        total_orders = 0.0
        total_sales = 0.0

        for _ in range(horizon):
            action = agent.get_action(state, exploration=False)
            next_state, reward, _, info = env.step(action)
            state = next_state
            episode_reward += float(reward)
            total_orders += float(info["inventory_post_order"] - info["inventory_pre_order"])
            total_sales += float(info["sales"])

        rewards.append(episode_reward)
        orders.append(total_orders / max(1, horizon))
        sales.append(total_sales / max(1, horizon))
        ending_inventory.append(float(state))

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "avg_order": float(np.mean(orders)) if orders else 0.0,
        "avg_sales": float(np.mean(sales)) if sales else 0.0,
        "avg_ending_inventory": float(np.mean(ending_inventory)) if ending_inventory else 0.0,
        "episodes": episodes,
        "horizon": horizon,
    }


def main() -> None:  # pragma: no cover
    config = parse_args()
    train_agent(config)


if __name__ == "__main__":  # pragma: no cover
    main()
