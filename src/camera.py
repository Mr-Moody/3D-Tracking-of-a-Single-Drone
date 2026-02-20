import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np


class Camera():
    """Camera with intrinsics, extrinsics, and projection for EKF measurement model."""

    def __init__(
        self,
        filepath: Optional[Path] = None,
        K: Optional[np.ndarray] = None,
        dist_coeff: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
    ):
        self.filepath = filepath
        self.capture = None
        self.current_frame = 0

        self.K = K if K is not None else np.eye(3)
        self.dist_coeff = dist_coeff if dist_coeff is not None else np.zeros(5)
        self.P = P  # 3x4 projection matrix; must be set for project/projection_jacobian

    def load_from_file(self) -> None:
        if self.filepath is None:
            raise ValueError("No video filepath set")
        
        self.capture = cv.VideoCapture(str(self.filepath))
        
        if not self.capture.isOpened():
            raise ValueError("Could not open video file")
        
        self.current_frame = 0

    def get_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video. Only one frame in memory at a time."""
        if self.capture is None:
            return None
        
        ret, frame = self.capture.read()
        
        if not ret:
            return None
        
        self.current_frame += 1
        
        return frame

    def release(self) -> None:
        """Release the video stream and reset the current frame to 0."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.current_frame = 0

    def project(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """Project 3D point (x,y,z) to 2D pixel (u,v) via P.

        [wu, wv, w]^T = P [x, y, z, 1]^T  =>  u = p0/p2, v = p1/p2
        """
        if self.P is None:
            raise ValueError("Projection matrix P not set")
        
        p = self.P @ np.append(np.asarray(point_3d, dtype=float).ravel()[:3], 1.0)
        w = p[2]
        
        if abs(w) < 1e-10:
            raise ValueError("Point behind or on camera plane (w == 0)")
        
        u = p[0] / w
        v = p[1] / w
        
        return float(u), float(v)

    def projection_jacobian(self, point_3d: np.ndarray) -> np.ndarray:
        """Jacobian of (u,v) with respect to (x,y,z) as a 2x3 matrix."""
        if self.P is None:
            raise ValueError("Projection matrix P not set")
        
        xyz = np.asarray(point_3d, dtype=float).ravel()[:3]
        
        p = self.P @ np.append(xyz, 1.0)
        w = p[2]
        
        if abs(w) < 1e-10:
            raise ValueError("Point behind or on camera plane (w == 0)")
        
        w2 = w * w
        
        P0 = self.P[0, :3]
        P1 = self.P[1, :3]
        P2 = self.P[2, :3]
        
        p0, p1 = p[0], p[1]
        
        du = (P0 * w - p0 * P2) / w2
        dv = (P1 * w - p1 * P2) / w2
        
        return np.vstack([du, dv])

    def undistort_point(self, u: float, v: float) -> Tuple[float, float]:
        """Undistort a 2D measurement of (u, v) to get the pixel coordinates (x, y)."""
        points = np.array([[[u, v]]], dtype=np.float32)
        
        K = self.K.astype(np.float64)
        
        dist_coeff = np.asarray(self.dist_coeff, dtype=np.float64)
        undistorted_points = cv.undistortPoints(points, K, dist_coeff, P=K)
        
        return float(undistorted_points[0, 0, 0]), float(undistorted_points[0, 0, 1])

    @classmethod
    def from_calibration_json(cls, calib_path: Path, P: np.ndarray, video_path: Optional[Path] = None) -> "Camera":
        """Create camera from calibration JSON and given projection matrix P (3x4)."""
        with open(calib_path, encoding="utf-8") as f:
            data = json.load(f)
            
        K = np.array(data["K-matrix"], dtype=np.float64)
        dist_coeff = np.array(data["distCoeff"], dtype=np.float64)
        
        return cls(filepath=video_path, K=K, dist_coeff=dist_coeff, P=np.asarray(P, dtype=np.float64))

    @classmethod
    def from_calibration_and_extrinsics(
        cls,
        calib_path: Path,
        camera_center: np.ndarray,
        look_at: Optional[np.ndarray] = None,
        video_path: Optional[Path] = None,
    ) -> "Camera":
        """Create camera from calibration JSON and camera center (world coords).

        Builds P = K [R|t] assuming camera at camera_center looks at look_at (default origin).
        """
        with open(calib_path, encoding="utf-8") as f:
            data = json.load(f)
            
        K = np.array(data["K-matrix"], dtype=np.float64)
        dist_coeff = np.array(data["distCoeff"], dtype=np.float64)
        
        C = np.asarray(camera_center, dtype=np.float64).ravel()[:3]
        
        ref = np.zeros(3) if look_at is None else np.asarray(look_at, dtype=np.float64).ravel()[:3]
        
        R, t = _rotation_and_translation_from_center(C, ref)
        Rt = np.hstack([R, t.reshape(-1, 1)])
        
        P = K @ Rt
        
        return cls(filepath=video_path, K=K, dist_coeff=dist_coeff, P=P)

    @classmethod
    def from_mvus_pkl(cls, pkl_path: Path, cam_idx: int, video_path: Optional[Path] = None) -> "Camera":
        """Create camera from mvus pipeline output (.pkl) containing cameras[i]['P']."""
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
            
        cams = result["cameras"]
        
        if cam_idx >= len(cams):
            raise IndexError(f"Camera index {cam_idx} out of range (have {len(cams)} cameras)")
        
        cam = cams[cam_idx]
        
        P = np.array(cam["P"], dtype=np.float64)
        K = np.array(cam["K"], dtype=np.float64)
        
        dist_coeff = np.array(cam.get("d", [0] * 5), dtype=np.float64)
        
        if len(dist_coeff) < 5:
            dist_coeff = np.resize(dist_coeff, 5)
            
        return cls(filepath=video_path, K=K, dist_coeff=dist_coeff, P=P)


def _rotation_and_translation_from_center(C: np.ndarray, look_at: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R (3x3) and t (3,) for camera at C looking at look_at."""
    forward = look_at - C
    norm = np.linalg.norm(forward)
    
    if norm < 1e-10:
        raise ValueError("Camera center coincides with look-at point")
    
    forward = forward / norm
    up = np.array([0.0, 0.0, 1.0])
    
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
        
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Camera: x right, y down, z into scene -> rows of R = [r, -u, forward]
    R = np.array([right, -up, forward], dtype=np.float64)
    t = -R @ C
    
    return R, t
