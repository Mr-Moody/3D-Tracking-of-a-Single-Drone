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
        fps: Optional[float] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        self.filepath = filepath
        self.capture = None
        self.current_frame = 0

        self.K = K if K is not None else np.eye(3)
        self.dist_coeff = dist_coeff if dist_coeff is not None else np.zeros(5)
        self.P = P  # 3x4 projection matrix; must be set for project/projection_jacobian
        self.fps = fps
        self.resolution = resolution  # (width, height) from calibration if available

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

    def estimate_distance_to_bbox(
        self, x1: float, y1: float, x2: float, y2: float, real_width_m: float
    ) -> Optional[float]:
        """Estimate distance in metres to an object of known width from its bounding box (pinhole model).

        Returns None if the camera is uncalibrated (K is identity-like or fx is 0).
        """
        fx = float(self.K[0, 0])

        if fx <= 0 or abs(fx - 1.0) < 1e-6:
            return None

        bbox_width_px = max(1.0, x2 - x1)

        return (fx * real_width_m) / bbox_width_px

    def back_project_to_3d(self, u: float, v: float, depth: float) -> Optional[np.ndarray]:
        """Back-project pixel (u, v) with given depth to 3D point in camera frame.

        Returns None if K is invalid (e.g. identity) or depth <= 0.
        """
        if depth <= 0:
            return None
        fx = float(self.K[0, 0])
        if fx <= 0 or abs(fx - 1.0) < 1e-6:
            return None
        ray = np.linalg.inv(self.K) @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(ray[2]) < 1e-10:
            return None
        scale = depth / ray[2]
        point_camera = scale * ray
        return point_camera

    def camera_to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """Transform 3D point from camera frame to world frame using P = K[R|t].

        Requires self.P to be set. World point X satisfies p_c = R @ X + t, so X = R.T @ (p_c - t).
        """
        if self.P is None:
            raise ValueError("Projection matrix P not set")
        Rt = np.linalg.inv(self.K) @ self.P
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        point_camera = np.asarray(point_camera, dtype=np.float64).ravel()[:3]
        return (R.T @ (point_camera - t)).reshape(3)

    @classmethod
    def from_calibration_json(
        cls,
        calib_path: Path,
        P: Optional[np.ndarray] = None,
        video_path: Optional[Path] = None,
    ) -> "Camera":
        """Create camera from calibration JSON. P (3x4) is optional; if present in JSON it is used."""
        with open(calib_path, encoding="utf-8") as f:
            data = json.load(f)

        K = np.array(data["K-matrix"], dtype=np.float64)
        dist_coeff = np.array(data["distCoeff"], dtype=np.float64)
        P_val = data.get("P")
        if P_val is not None:
            P_use = np.asarray(P_val, dtype=np.float64)
        else:
            P_use = np.asarray(P, dtype=np.float64) if P is not None else None

        fps = data.get("fps")
        if fps is not None:
            fps = float(fps)
        res = data.get("resolution")
        if res is not None and len(res) >= 2:
            resolution = (int(res[0]), int(res[1]))
        else:
            resolution = None

        return cls(
            filepath=video_path,
            K=K,
            dist_coeff=dist_coeff,
            P=P_use,
            fps=fps,
            resolution=resolution,
        )

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
    def from_calibration_folder(
        cls,
        calibration_root: Path,
        camera_type: str,
        video_path: Optional[Path] = None,
        P: Optional[np.ndarray] = None,
    ) -> "Camera":
        """Create camera from central calibration folder: calibration_root/camera_type/<camera_type>.json or first .json in folder."""
        folder = calibration_root / camera_type
        if not folder.is_dir():
            raise FileNotFoundError(f"Calibration folder not found: {folder}")

        preferred = folder / f"{camera_type}.json"
        if preferred.exists():
            calib_path = preferred
        else:
            jsons = sorted(folder.glob("*.json"))
            if not jsons:
                raise FileNotFoundError(f"No .json file in calibration folder: {folder}")
            calib_path = jsons[0]

        return cls.from_calibration_json(calib_path, P=P, video_path=video_path)

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
