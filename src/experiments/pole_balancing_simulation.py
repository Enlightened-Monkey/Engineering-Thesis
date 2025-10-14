"""Visualise a trained pole-balancing agent as a 10-second GIF."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np

if TYPE_CHECKING:
    from .pole_balancing_training import EnvironmentConfig as EnvironmentConfigType
else:
    EnvironmentConfigType = Any

try:
    from .pole_balancing_training import EnvironmentConfig, build_env
    from ..algorithms.qh_qlearning import QHQLearning
except ImportError:  # pragma: no cover - support execution via `python path/to/script.py`
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    EXP_ROOT = SRC_ROOT / "experiments"
    for candidate in (REPO_ROOT, SRC_ROOT, EXP_ROOT):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    import pole_balancing_training as _pb_training  # type: ignore

    EnvironmentConfig = _pb_training.EnvironmentConfig  # type: ignore[attr-defined]
    build_env = _pb_training.build_env  # type: ignore[attr-defined]
    from algorithms.qh_qlearning import QHQLearning  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a GIF of a trained pole-balancing agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained agent .npz snapshot")
    parser.add_argument("--output", type=Path, default=Path("data/plots/pole_balancing_agent.gif"), help="Destination GIF path")
    parser.add_argument("--duration", type=float, default=10.0, help="Animation duration in seconds")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for the GIF; defaults to environment step frequency")
    parser.add_argument("--dpi", type=int, default=120, help="Output resolution (dots per inch)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible reset noise")
    parser.add_argument("--env-json", type=Path, default=None, help="Optional JSON with environment config (falls back to metadata)")
    parser.add_argument("--title", type=str, default="Pole Balancing (QH Q-Learning)", help="Title shown on the animation")
    parser.add_argument("--no-blit", action="store_true", help="Disable blitting (useful on some systems)")
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be strictly positive")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be a positive integer")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    return args


def _extract_env_config(source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "config" in source and isinstance(source["config"], dict):
        config_block = source["config"]
        if "environment" in config_block and isinstance(config_block["environment"], dict):
            return config_block["environment"]
    if "environment" in source and isinstance(source["environment"], dict):
        return source["environment"]
    return None


def _load_env_config(metadata: Optional[Dict[str, Any]], env_json_path: Optional[Path]) -> EnvironmentConfigType:
    env_dict: Optional[Dict[str, Any]] = None

    if metadata is not None:
        env_dict = _extract_env_config(metadata)

    if env_dict is None and env_json_path is not None:
        with env_json_path.open("r", encoding="utf-8") as fh:
            env_source = json.load(fh)
        env_dict = _extract_env_config(env_source)

    if env_dict is None:
        print("WARNING: Falling back to default environment configuration; behaviour may differ from training run.")
        return EnvironmentConfig()

    allowed_keys = {field.name for field in fields(EnvironmentConfig)}
    filtered = {k: env_dict[k] for k in allowed_keys if k in env_dict}
    return EnvironmentConfig(**filtered)


def _capture_frame(env, info: Optional[Dict[str, Any]], time_elapsed: float) -> Dict[str, float | str | None]:
    if info is None:
        x, _, theta, _ = env._continuous_state  # type: ignore[attr-defined]
        length = env._length  # type: ignore[attr-defined]
        terminated = None
    else:
        x = info["x"]
        theta = info["theta"]
        length = info["length"]
        terminated = info.get("terminated_reason")
    return {
        "x": float(x),
        "theta": float(theta),
        "length": float(length),
        "time": float(time_elapsed),
        "terminated": terminated,
    }


def _rollout(agent: QHQLearning, env: Any, duration: float, fps: Optional[int], seed: Optional[int]) -> Tuple[List[Dict[str, Any]], int]:
    if seed is not None:
        np.random.seed(seed)

    frames: List[Dict[str, Any]] = []
    state = env.reset()
    time_elapsed = 0.0
    frames.append(_capture_frame(env, None, time_elapsed))

    if fps is None:
        fps = max(1, int(round(1.0 / env.time_step)))

    target_frames = max(1, int(round(duration * fps)))
    dt = 1.0 / fps

    # Align control frequency with environment time step
    steps_per_frame = max(1, int(round(dt / env.time_step)))
    total_steps = target_frames * steps_per_frame

    for frame_idx in range(target_frames):
        info: Optional[Dict[str, Any]] = None
        for _ in range(steps_per_frame):
            action = agent.get_action(state, exploration=False)
            next_state, _, done, info = env.step(action)
            state = next_state
            time_elapsed = info["time_elapsed"] if info and "time_elapsed" in info else env._time_elapsed
            if done:
                break
        frames.append(_capture_frame(env, info, time_elapsed))
        if done:
            last_frame = dict(frames[-1])
            while len(frames) < target_frames + 1:
                frames.append(dict(last_frame))
            break

    return frames, fps


def _animate(frames: List[Dict[str, Any]], env: Any, output_path: Path, fps: int, dpi: int, title: str, disable_blit: bool) -> Path:
    cart_width = 0.4
    cart_height = 0.25
    axle_height = cart_height / 2
    rail_y = 0.0

    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.set_xlim(env.min_x - 1.0, env.max_x + 1.0)
    max_pole_height = env.length_range[1] + 0.5
    ax.set_ylim(-0.5, max_pole_height + 0.5)
    ax.set_xlabel("Cart position")
    ax.set_ylabel("Pole height")
    ax.set_title(title)
    ax.axhline(rail_y, color="black", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")

    cart = patches.FancyBboxPatch((0, rail_y), cart_width, cart_height, boxstyle="round,pad=0.02", color="#1976D2")
    pole_line, = ax.plot([], [], linewidth=3, color="tab:red")
    axle = patches.Circle((0, rail_y + axle_height), radius=0.03, color="black")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    status_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, color="tab:red")

    ax.add_patch(cart)
    ax.add_patch(axle)

    def _update(frame: Dict[str, Any]):
        x = frame["x"]
        theta = frame["theta"]
        length = frame["length"]
        t = frame["time"]
        terminated = frame.get("terminated")

        cart.set_x(x - cart_width / 2)
        axle.center = (x, rail_y + axle_height)

        pole_x_end = x + length * np.sin(theta)
        pole_y_end = rail_y + axle_height + length * np.cos(theta)
        pole_line.set_data([x, pole_x_end], [rail_y + axle_height, pole_y_end])

        time_text.set_text(f"t = {t:5.2f} s")
        status_text.set_text("" if not terminated else f"terminated: {terminated}")

        return cart, pole_line, axle, time_text, status_text

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=1000 / fps,
        blit=not disable_blit,
        repeat=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    return output_path


def main() -> None:
    args = _parse_args()

    agent, metadata = QHQLearning.load(args.model, return_metadata=True)
    env_config = _load_env_config(metadata, args.env_json)
    env = build_env(env_config)

    frames, fps = _rollout(agent, env, duration=args.duration, fps=args.fps, seed=args.seed)
    output_path = _animate(frames, env, args.output, fps=fps, dpi=args.dpi, title=args.title, disable_blit=args.no_blit)

    print(f"Saved animation to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
