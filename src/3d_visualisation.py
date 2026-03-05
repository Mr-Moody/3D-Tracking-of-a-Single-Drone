import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_path_route(path: Path | str) -> np.ndarray:
    """Load positions (x, y, z) from a path_route.json file as an (N, 3) array."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"path_route file not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError(f"No records in path route file: {path}")

    positions = []

    for rec in records:
        pos = rec.get("position")
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float64).ravel()[:3]
        if arr.shape[0] == 3:
            positions.append(arr)

    if not positions:
        raise ValueError(f"No valid position entries in path route file: {path}")

    return np.stack(positions, axis=0)


def _load_rtk_points(rtk_txt: Path | str) -> Optional[np.ndarray]:
    """Load RTK ground-truth trajectory from trajectory/rtk.txt as an (M, 3) array.

    Returns None when the file is missing or no valid points exist.
    """
    rtk_path = Path(rtk_txt)

    if not rtk_path.exists():
        return None

    points: list[np.ndarray] = []

    with open(rtk_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            try:
                pts = np.array(
                    [float(parts[0]), float(parts[1]), float(parts[2])],
                    dtype=np.float64,
                )
            except ValueError:
                continue

            points.append(pts)

    if not points:
        return None

    return np.stack(points, axis=0)


def plot_trajectory_3d(
    dataset_dir: Path | str,
    show_ground_truth: bool = True,
    elev: float = 30.0,
    azim: float = -60.0,
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """Visualise drone trajectory in 3D using saved path_route.json and optional RTK.

    Args:
        dataset_dir: Dataset folder (e.g. drone-tracking-datasets/dataset1).
        show_ground_truth: Overlay RTK ground truth when available.
        elev: 3D elevation angle (degrees).
        azim: 3D azimuth angle (degrees).
        save_path: Optional path to save the figure (e.g. PNG).
        show: If True, display the figure; otherwise close it after saving.
    """
    dataset_dir = Path(dataset_dir)
    path_route_path = dataset_dir / "path_route.json"
    positions = _load_path_route(path_route_path)  # (N, 3)

    gt = None

    if show_ground_truth:
        rtk_path = dataset_dir / "trajectory" / "rtk.txt"
        gt = _load_rtk_points(rtk_path)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Estimated EKF trajectory
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        label="EKF estimate",
        color="tab:blue",
        linewidth=2.0,
    )

    # Start and end markers
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        color="green",
        s=40,
        label="Start",
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        color="red",
        s=40,
        label="End",
    )

    # Optional RTK ground truth overlay
    if gt is not None:
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            gt[:, 2],
            label="RTK ground truth",
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # Approximate equal aspect ratio so geometry is not distorted
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    ax.set_box_aspect(ranges)

    ax.view_init(elev=elev, azim=azim)
    ax.legend()
    ax.set_title(f"Drone trajectory in 3D - {dataset_dir.name}")

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise 3D drone trajectory from path_route.json and optional RTK.",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        required=True,
        help="Dataset index (e.g. 1 for dataset1).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="drone-tracking-datasets",
        help="Root directory containing dataset folders (default: drone-tracking-datasets).",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Disable RTK ground-truth overlay even if available.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g. trajectory_3d.png).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_dir = Path(args.root) / f"dataset{args.dataset}"
    plot_trajectory_3d(
        dataset_dir=dataset_dir,
        show_ground_truth=not args.no_gt,
        save_path=args.save,
        show=True,
    )
