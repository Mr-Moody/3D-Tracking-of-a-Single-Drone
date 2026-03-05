import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import List, Optional, cast
import time

import torch

from src.window import Window
from src.camera import Camera
from src.drone_detection import DroneDetector
from src.fusion import fuse_positions
from src.kalman_filter import DroneTrackerEKF
from src.path_route import PathRoute

PATH = Path("drone-tracking-datasets")
CALIBRATION_ROOT = PATH / "calibration"
DRONE_REAL_WIDTH_M = 0.3  # Real-world width of drone in metres (for distance estimation)


def load_cameras_for_dataset(dataset_dir: Path, calibration_root: Path = CALIBRATION_ROOT) -> List[Camera]:
    """Load one Camera per entry in dataset_dir/cameras.txt using calibration_root/<camera_type>/.

    cameras.txt format: one line per camera, e.g. "cam0 - iphone6" (cam_id and camera type).
    Video path is dataset_dir/<cam_id>.mp4; calibration is calibration_root/<camera_type>/<camera_type>.json or first .json.
    """
    cameras_txt = dataset_dir / "cameras.txt"
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Dataset cameras file not found: {cameras_txt}")

    cameras: List[Camera] = []

    with open(cameras_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("-", 1)

            if len(parts) != 2:
                continue

            cam_id = parts[0].strip()
            camera_type = parts[1].strip()
            video_path = dataset_dir / f"{cam_id}.mp4"

            cam = Camera.from_calibration_folder(calibration_root, camera_type, video_path=video_path, P=None)

            cameras.append(cam)

    if not cameras:
        raise ValueError(f"No cameras parsed from {cameras_txt}")

    return cameras


def load_ground_truth_rtk(dataset_dir: Path) -> Optional[List[np.ndarray]]:
    """Load trajectory/rtk.txt if present; return list of (x,y,z) per frame or None."""
    rtk_path = dataset_dir / "trajectory" / "rtk.txt"

    if not rtk_path.exists():
        return None
    points = []

    with open(rtk_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 3:
                points.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64))

    return points if points else None


def _verify_gpu() -> None:
    """Print GPU status and verify CUDA is available."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name} (CUDA available)")
    else:
        print("WARNING: GPU not available, using CPU")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera drone tracking with 3D fusion and Kalman filter")
    parser.add_argument("--dataset", type=int, default=1, help="Dataset number (e.g. 1 -> dataset1)")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI window (no image display).",
    )
    args = parser.parse_args()

    _verify_gpu()

    dataset_dir = PATH / f"dataset{args.dataset}"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    cameras = load_cameras_for_dataset(dataset_dir)

    for cam in cameras:
        cam.load_from_file()

    # Estimate total frame count across all cameras for progress logging
    total_frames: Optional[int] = None
    if cameras and all(cam.capture is not None for cam in cameras):
        counts = []
        for cam in cameras:
            n = int(cast(int, cam.capture.get(cv.CAP_PROP_FRAME_COUNT)))  # type: ignore[union-attr]
            if n > 0:
                counts.append(n)
        if counts:
            total_frames = min(counts)

    window = None if args.no_gui else Window()
    detector = DroneDetector(
        model_path=Path("models/yolo11s.pt"),
        model_size="s",
        use_sahi=False,
        use_motion_roi=True,
        slice_height=896,
        slice_width=896,
        imgsz=320,
        half=True,
    )
    ekf = DroneTrackerEKF()
    path_route = PathRoute()

    gt_rtk = load_ground_truth_rtk(dataset_dir)
    cached_per_camera: List[List[tuple]] = [[] for _ in cameras]
    frame_index = 0
    prev_time = time.time()
    detect_every_n = 1

    while True:
        frames = [cam.get_frame() for cam in cameras]

        if any(f is None for f in frames):
            break
        frames = cast(List[np.ndarray], [f for f in frames if f is not None])

        if len(frames) != len(cameras):
            break

        # Progress logging in console
        if total_frames is not None:
            print(f"Processing frame {frame_index + 1}/{total_frames}", end="\r", flush=True)
        else:
            print(f"Processing frame {frame_index + 1}", end="\r", flush=True)

        positions_per_camera: List[Optional[np.ndarray]] = [None] * len(cameras)
        first_cam_detections: List[tuple] = []

        for i, (cam, frame) in enumerate(zip(cameras, frames)):
            run_detector = detect_every_n <= 1 or (frame_index % detect_every_n) == 0
            detections = detector.detect(frame) if run_detector else []

            if detections:
                cached_per_camera[i] = detections
            else:
                detections = cached_per_camera[i]

            if detections:
                best = max(detections, key=lambda d: d[4])
                x1, y1, x2, y2, conf = best

                if i == 0:
                    first_cam_detections = detections

                u = (x1 + x2) / 2
                v = (y1 + y2) / 2

                dist = cam.estimate_distance_to_bbox(x1, y1, x2, y2, DRONE_REAL_WIDTH_M)

                if dist is not None:
                    p_cam = cam.back_project_to_3d(u, v, dist)

                    if p_cam is not None:
                        positions_per_camera[i] = p_cam

        fused = fuse_positions(positions_per_camera, cameras)

        now = time.time()
        dt = now - prev_time
        prev_time = now

        if dt <= 0:
            dt = 1.0 / 30.0

        ekf.predict(dt)

        if fused is not None:
            ekf.update_3d(fused)
            path_route.append(
                now,
                ekf.get_position(),
                ekf.get_velocity(),
                ekf.get_acceleration(),
            )

        if not args.no_gui:
            display_frame = frames[0].copy()

            for detection in first_cam_detections:
                x1, y1, x2, y2, confidence = detection

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                cv.rectangle(display_frame, pt1, pt2, (0, 0, 255), 2)
                cv.putText(display_frame, f"{confidence:.2f}", pt1, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                distance = cameras[0].estimate_distance_to_bbox(x1, y1, x2, y2, DRONE_REAL_WIDTH_M)
                label_y = int(y2) + 20

                if distance is not None:
                    cv.putText(display_frame, f"{distance:.1f} m", (int(x1), label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv.putText(display_frame, "N/A", (int(x1), label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            pos = ekf.get_position()
            vel = ekf.get_velocity()
            acc = ekf.get_acceleration()

            cv.putText(display_frame, f"pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(display_frame, f"vel: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv.putText(display_frame, f"acc: [{acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}]", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if gt_rtk is not None and frame_index < len(gt_rtk):
                err = float(np.linalg.norm(pos - gt_rtk[frame_index]))
                cv.putText(display_frame, f"GT err: {err:.3f} m", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if window is not None:
                window.show_image(display_frame)
            cv.pollKey()

        frame_index += 1

    path_route.save(dataset_dir / "path_route.json")

    # Ensure progress line ends cleanly
    print()

    if window is not None:
        window.close()

    for cam in cameras:
        cam.release()

    print(f"Saved path_route.json with {len(path_route)} samples.")


if __name__ == "__main__":
    main()
