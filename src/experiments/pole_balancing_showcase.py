"""Interactive showcase for the pole balancing training GIF."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import tkinter as tk
from PIL import Image, ImageSequence, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GIF_PATH = REPO_ROOT / "data/plots/pole_balancing_100k.gif"


def load_gif_frames(path: Path) -> Tuple[List[Image.Image], List[int]]:
    """Load GIF frames as PIL images together with their durations (in ms)."""
    if not path.exists():
        raise FileNotFoundError(path)

    frames: List[Image.Image] = []
    durations: List[int] = []

    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            frames.append(frame.convert("RGBA"))
            durations.append(max(20, int(frame.info.get("duration", 100))))

    if not frames:
        raise ValueError(f"No frames could be read from GIF: {path}")

    return frames, durations


def prepare_tk_frames(
    pil_frames: List[Image.Image],
    root: tk.Misc,
    scale: float,
) -> List[ImageTk.PhotoImage]:
    """Convert PIL frames into Tk-compatible images, applying optional scaling."""
    if scale <= 0:
        raise ValueError("scale must be positive")

    tk_frames: List[ImageTk.PhotoImage] = []
    for frame in pil_frames:
        if scale != 1.0:
            width = int(frame.width * scale)
            height = int(frame.height * scale)
            scaled = frame.resize((width, height), Image.NEAREST)
        else:
            scaled = frame
        tk_frames.append(ImageTk.PhotoImage(scaled, master=root))
    return tk_frames


def play_animation(frames: List[ImageTk.PhotoImage], durations: List[int], root: tk.Tk) -> None:
    """Start the Tk main loop and animate the frames indefinitely."""
    label = tk.Label(root)
    label.pack(padx=12, pady=12)

    footer = tk.Label(root, text="Press Esc or close the window to exit.")
    footer.pack(pady=(0, 12))

    def update(frame_idx: int = 0) -> None:
        label.configure(image=frames[frame_idx])
        next_idx = (frame_idx + 1) % len(frames)
        root.after(durations[frame_idx], update, next_idx)

    update()
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play back the pole balancing training GIF in a simple Tk window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF_PATH, help="Path to the GIF file to play")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scaling factor for the GIF frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gif_path = args.gif
    if not gif_path.is_absolute():
        gif_path = (Path.cwd() / gif_path).resolve()

    pil_frames, durations = load_gif_frames(gif_path)

    root = tk.Tk()
    root.title(f"Pole Balancing Showcase – {gif_path.name}")
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    tk_frames = prepare_tk_frames(pil_frames, root, scale=float(args.scale))

    play_animation(tk_frames, durations, root)


if __name__ == "__main__":
    main()
