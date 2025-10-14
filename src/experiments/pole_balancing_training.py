"""Command-line tool for training QH Q-Learning on the pole-balancing environment."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from ..algorithms.qh_qlearning import QHQLearning
    from ..models.mdp_environments import PoleBalancingMDP
except ImportError:  # pragma: no cover - support execution via `python path/to/script.py`
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    for candidate in (REPO_ROOT, SRC_ROOT):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))
    from algorithms.qh_qlearning import QHQLearning  # type: ignore
    from models.mdp_environments import PoleBalancingMDP  # type: ignore


@dataclass
class EnvironmentConfig:
    min_x: float = -2.4
    max_x: float = 2.4
    max_speed: float = 2.5
    force_mag: float = 10.0
    wind_force_max: float = 3.0
    wind_turbulence: float = 0.5
    time_step: float = 0.02
    max_time: float = 10.0
    angle_reward_threshold: float = math.radians(12)
    angle_failure: float = math.radians(45)
    length_min: float = 0.5
    length_max: float = 2.0
    mass_per_meter: float = 0.5
    n_position_bins: int = 7
    n_velocity_bins: int = 7
    n_angle_bins: int = 11
    n_ang_velocity_bins: int = 7
    n_length_bins: int = 3
    fall_penalty: float = 10.0
    success_bonus: float = 2.0


@dataclass
class TrainingConfig:
    episodes: int = 100_000
    eval_episodes: int = 25
    sigma: float = 0.7
    gamma: float = 0.97
    alpha: float = 0.08
    epsilon: float = 0.2
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.995
    seed: Optional[int] = None
    results_dir: Path = Path("data/results")
    model_dir: Path = Path("data/models")
    save_details: bool = True
    save_model: bool = True
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def parse_args() -> TrainingConfig:
    env_defaults = EnvironmentConfig()

    parser = argparse.ArgumentParser(
        description="Train quasi-hyperbolic Q-learning on the pole-balancing MDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=TrainingConfig.eval_episodes, help="Evaluation rollouts after training")
    parser.add_argument("--sigma", type=float, default=TrainingConfig.sigma, help="Present-bias parameter σ")
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma, help="Exponential discount γ")
    parser.add_argument("--alpha", type=float, default=TrainingConfig.alpha, help="Learning rate α")
    parser.add_argument("--epsilon", type=float, default=TrainingConfig.epsilon, help="Initial ε for exploration")
    parser.add_argument("--min-epsilon", type=float, default=TrainingConfig.min_epsilon, help="Minimum ε during decay")
    parser.add_argument("--epsilon-decay", type=float, default=TrainingConfig.epsilon_decay, help="Multiplicative decay for ε every 50 episodes")
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed, help="Random seed for reproducibility")
    parser.add_argument("--results-dir", type=Path, default=TrainingConfig.results_dir, help="Directory for saving training artefacts")
    parser.add_argument("--model-dir", type=Path, default=TrainingConfig.model_dir, help="Directory for saving trained agent snapshots")
    parser.add_argument("--no-save", action="store_true", help="Skip saving JSON summary to disk")
    parser.add_argument("--no-save-model", action="store_true", help="Skip saving trained agent weights")

    parser.add_argument("--force-mag", type=float, default=env_defaults.force_mag, help="Magnitude of cart push force")
    parser.add_argument("--max-speed", type=float, default=env_defaults.max_speed, help="Maximum absolute cart speed")
    parser.add_argument("--wind-force-max", type=float, default=env_defaults.wind_force_max, help="Maximum absolute wind force")
    parser.add_argument("--wind-turbulence", type=float, default=env_defaults.wind_turbulence, help="Wind noise standard deviation per step")
    parser.add_argument("--length-min", type=float, default=env_defaults.length_min, help="Minimum pole length")
    parser.add_argument("--length-max", type=float, default=env_defaults.length_max, help="Maximum pole length")
    parser.add_argument("--max-time", type=float, default=env_defaults.max_time, help="Maximum episode length in seconds")
    parser.add_argument("--angle-threshold-deg", type=float, default=math.degrees(env_defaults.angle_reward_threshold), help="Angle (deg) for full upright reward")
    parser.add_argument("--angle-failure-deg", type=float, default=math.degrees(env_defaults.angle_failure), help="Angle (deg) at which the pole is considered fallen")

    args = parser.parse_args()

    env_config = EnvironmentConfig(
        force_mag=args.force_mag,
        max_speed=args.max_speed,
        wind_force_max=args.wind_force_max,
        wind_turbulence=args.wind_turbulence,
        length_min=args.length_min,
        length_max=args.length_max,
        max_time=args.max_time,
        angle_reward_threshold=math.radians(args.angle_threshold_deg),
        angle_failure=math.radians(args.angle_failure_deg),
    )

    return TrainingConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        sigma=args.sigma,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        results_dir=args.results_dir,
        model_dir=args.model_dir,
        save_details=not args.no_save,
        save_model=not args.no_save_model,
        environment=env_config,
    )


def evaluate_policy(agent: QHQLearning, env: PoleBalancingMDP, episodes: int) -> Dict[str, float | List[Dict[str, float | str]]]:
    stats: List[Dict[str, float | str]] = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info: Dict[str, float | str] = {}

        while not done:
            action = agent.get_action(state, exploration=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += float(reward)
            steps += 1
            last_info = info

        termination = (last_info.get("terminated_reason") if last_info else None) or "timeout"
        time_elapsed = float(last_info.get("time_elapsed", steps * env.time_step)) if last_info else steps * env.time_step

        stats.append(
            {
                "reward": total_reward,
                "duration_s": time_elapsed,
                "steps": float(steps),
                "termination": termination,
                "length": float(last_info.get("length", env._length)) if last_info else float(env._length),
            }
        )

    rewards = [s["reward"] for s in stats]
    durations = [s["duration_s"] for s in stats]
    fall_rate = sum(1 for s in stats if s["termination"] == "fall") / max(1, len(stats))

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "avg_duration_s": float(np.mean(durations)) if durations else 0.0,
        "fall_rate": float(fall_rate),
        "episodes": stats,
    }


def decay_epsilon(agent: QHQLearning, config: TrainingConfig, episode: int) -> None:
    if episode % 50 == 0 and episode > 0:
        agent.epsilon = max(config.min_epsilon, agent.epsilon * config.epsilon_decay)


def build_env(cfg: EnvironmentConfig) -> PoleBalancingMDP:
    return PoleBalancingMDP(
        min_x=cfg.min_x,
        max_x=cfg.max_x,
        max_speed=cfg.max_speed,
        force_mag=cfg.force_mag,
        wind_force_max=cfg.wind_force_max,
        wind_turbulence=cfg.wind_turbulence,
        time_step=cfg.time_step,
        max_time=cfg.max_time,
        angle_reward_threshold=cfg.angle_reward_threshold,
        angle_failure=cfg.angle_failure,
        length_range=(cfg.length_min, cfg.length_max),
        mass_per_meter=cfg.mass_per_meter,
        n_position_bins=cfg.n_position_bins,
        n_velocity_bins=cfg.n_velocity_bins,
        n_angle_bins=cfg.n_angle_bins,
        n_ang_velocity_bins=cfg.n_ang_velocity_bins,
        n_length_bins=cfg.n_length_bins,
        fall_penalty=cfg.fall_penalty,
        success_bonus=cfg.success_bonus,
    )


def train(config: TrainingConfig) -> Dict:
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
        done = False
        cumulative_reward = 0.0

        while not done:
            action = agent.get_action(state, exploration=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            cumulative_reward += float(reward)

        episode_rewards.append(cumulative_reward)
        decay_epsilon(agent, config, episode)

        if (episode + 1) % max(1, config.episodes // 10) == 0:
            print(f"Episode {episode + 1}/{config.episodes}: reward={cumulative_reward:.2f}, epsilon={agent.epsilon:.3f}")

    training_time = time.time() - start_time

    evaluation_env = build_env(config.environment)  # fresh randomness for evaluation
    evaluation = evaluate_policy(agent, evaluation_env, config.eval_episodes)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    config_dict = asdict(config)
    config_dict["results_dir"] = str(config.results_dir)
    config_dict["model_dir"] = str(config.model_dir)
    config_dict["environment"] = asdict(config.environment)

    model_path_str: Optional[str] = None
    if config.save_model:
        metadata = {
            "timestamp": timestamp,
            "config": config_dict,
            "summary": {
                "final_epsilon": agent.epsilon,
                "avg_reward": evaluation["avg_reward"],
                "avg_duration_s": evaluation["avg_duration_s"],
                "fall_rate": evaluation["fall_rate"],
            },
        }

        config.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = config.model_dir / f"pole_balancing_qh_{timestamp}.npz"
        agent.save(model_path, metadata=metadata)
        model_path_str = str(model_path)
        print(f"Saved trained agent to {model_path}")

    results = {
        "config": config_dict,
        "training": {
            "episodes": config.episodes,
            "training_time_s": training_time,
            "episode_rewards": episode_rewards,
            "final_epsilon": agent.epsilon,
            "final_policy": agent.get_policy().tolist(),
            "final_values": agent.get_value_function().tolist(),
        },
        "evaluation": evaluation,
        "artifacts": {
            "model_path": model_path_str,
        },
    }

    if config.save_details:
        config.results_dir.mkdir(parents=True, exist_ok=True)
        save_path = config.results_dir / f"pole_balancing_qh_{timestamp}.json"
        with save_path.open("w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        print(f"Saved training summary to {save_path}")

    print("\nEvaluation summary:")
    print(f"  Avg reward     : {evaluation['avg_reward']:.3f} ± {evaluation['std_reward']:.3f}")
    print(f"  Avg duration   : {evaluation['avg_duration_s']:.3f}s")
    print(f"  Fall rate      : {evaluation['fall_rate'] * 100:.1f}%")

    return results


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
