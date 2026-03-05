"""Fuse per-camera 3D position estimates into a single world-frame position."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.camera import Camera

def fuse_positions(
    positions_per_camera: List[Optional[np.ndarray]],
    cameras: List["Camera"],
) -> Optional[np.ndarray]:
    """Fuse per-camera 3D estimates into one global position in world frame.

    When at least one camera has P (world-to-camera projection), collect world-frame
    positions from those cameras and return their mean. When no camera has P, use
    cam0's position only (treat cam0 frame as world).
    """
    if not cameras or len(positions_per_camera) != len(cameras):
        return None

    has_any_p = any(cam.P is not None for cam in cameras)
    if has_any_p:

        world_positions = []
        for pos, cam in zip(positions_per_camera, cameras):
            if pos is not None and cam.P is not None:
                try:
                    world_positions.append(cam.camera_to_world(pos))
                except (ValueError, np.linalg.LinAlgError):
                    pass

        if not world_positions:
            return None

        return np.mean(world_positions, axis=0).reshape(3)

    if positions_per_camera[0] is not None:
        return np.asarray(positions_per_camera[0], dtype=np.float64).ravel()[:3].copy()

    return None
