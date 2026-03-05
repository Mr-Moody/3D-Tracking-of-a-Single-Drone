"""Store drone trajectory (path route) with timestamp, position, velocity, acceleration in JSON."""

import json
from pathlib import Path
from typing import List, Union

import numpy as np


class PathRoute:
    """Holds a list of trajectory samples for JSON export."""

    def __init__(self) -> None:
        self._records: List[dict] = []

    def append(
        self,
        timestamp: float,
        position: Union[List[float], np.ndarray],
        velocity: Union[List[float], np.ndarray],
        acceleration: Union[List[float], np.ndarray],
    ) -> None:
        """Add one record. Converts numpy arrays to lists for JSON."""
        self._records.append({
            "timestamp": float(timestamp),
            "position": np.asarray(position, dtype=np.float64).ravel()[:3].tolist(),
            "velocity": np.asarray(velocity, dtype=np.float64).ravel()[:3].tolist(),
            "acceleration": np.asarray(acceleration, dtype=np.float64).ravel()[:3].tolist(),
        })

    def to_json(self) -> str:
        """Return JSON string of the list of records."""
        return json.dumps(self._records, indent=2)

    def save(self, path: Path) -> None:
        """Write the trajectory to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def __len__(self) -> int:
        return len(self._records)
