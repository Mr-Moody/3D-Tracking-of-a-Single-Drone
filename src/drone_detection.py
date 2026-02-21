"""Drone detection using YOLO11, SAHI slicing, and motion-informed ROI.

Supports:
- YOLO11 S/M (or custom drone-trained weights from Det-Fly, Real World Drone Dataset).
- SAHI: sliced inference for small objects in high-res frames (e.g. 4K).
- MOG2: motion-based ROI to run detection only on moving regions (fixed cameras).

Training recommendations:
- Fine-tune YOLO11s/m on drone-specific datasets (Det-Fly, Real World Drone Dataset).
- Include negative samples (birds, planes, balloons) to reduce false positives.
- Prefer S or M; avoid Nano (too weak at ~300 m) and X-Large (too slow for multi-camera FPS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _extract_yolo_detections(results) -> list[tuple[float, float, float, float, float]]:
    """Convert ultralytics results to [(x1, y1, x2, y2, conf), ...]."""
    detections = []
    
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections.append((float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), conf))
            
    return detections


class DroneDetector():
    """Drone detector with YOLO11, optional SAHI slicing and MOG2 motion ROI."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_size: Literal["s", "m"] = "s",
        use_sahi: bool = False,
        use_motion_roi: bool = False,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.15,
        overlap_width_ratio: float = 0.15,
        sahi_threshold: int = 1280,
        min_motion_area: int = 200,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        imgsz: int = 480,
        half: bool = True,
    ):
        """Create a drone detector.

        Args:
            model_path: Path to custom weights (.pt). If None, uses yolo11s.pt or yolo11m.pt.
            model_size: 's' or 'm' when model_path is None.
            use_sahi: Enable SAHI sliced inference for small objects.
            use_motion_roi: Use MOG2 to restrict detection to moving regions (fixed cameras).
            slice_height, slice_width: SAHI tile size (default 640).
            overlap_height_ratio, overlap_width_ratio: SAHI overlap (0.1-0.2).
            sahi_threshold: Use SAHI when frame width or height > this (even if use_sahi=False).
            min_motion_area: Min contour area (pixels^2) for MOG2 ROIs.
            conf_threshold: YOLO confidence threshold.
            iou_threshold: YOLO IoU threshold.
            device: 'cuda:0', 'cpu', etc.
            imgsz: YOLO input size (smaller = faster, e.g. 480 or 416).
            half: Use FP16 on GPU for faster inference (ignored on CPU).
        """
        self.use_sahi = use_sahi
        self.use_motion_roi = use_motion_roi
        self.imgsz = imgsz
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.sahi_threshold = sahi_threshold
        self.min_motion_area = min_motion_area
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device is not None else str(DEVICE)
        self._use_half = half and "cuda" in str(self.device).lower()
        self._sahi_model = None

        if model_path is not None:
            path = str(Path(model_path).expanduser().resolve())
        else:
            path = f"yolo11{model_size}.pt"
            
        self.model_path = path
        self.yolo_model = YOLO(path)
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False,
        )
        self._motion_roi_fallback_counter = 0

    def _get_motion_rois(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) rectangles for moving regions."""
        fgmask = self._mog2.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        
        for c in contours:
            area = cv2.contourArea(c)
            
            if area < self.min_motion_area:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            rois.append((x, y, w, h))
            
        return rois

    def _merge_overlapping_rois(self, rois: list[tuple[int, int, int, int]], pad: int = 64) -> list[tuple[int, int, int, int]]:
        """Merge overlapping ROIs and pad to minimum size for YOLO."""
        if not rois:
            return []
        
        merged = []
        
        for x, y, w, h in rois:
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = x + w + pad
            y2 = y + h + pad
            
            merged.append((x1, y1, x2 - x1, y2 - y1))
            
        # Merge overlapping
        if len(merged) <= 1:
            return merged
        
        result = [merged[0]]
        
        for x, y, w, h in merged[1:]:
            x2, y2 = x + w, y + h
            combined = False
            
            for i, (rx, ry, rw, rh) in enumerate(result):
                rx2, ry2 = rx + rw, ry + rh
                
                if not (x2 < rx or x > rx2 or y2 < ry or y > ry2):
                    nx = min(x, rx)
                    ny = min(y, ry)
                    nx2 = max(x2, rx2)
                    ny2 = max(y2, ry2)
                    result[i] = (nx, ny, nx2 - nx, ny2 - ny)
                    combined = True
                    break
                
            if not combined:
                result.append((x, y, w, h))
                
        return result

    def _run_full_frame_yolo(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Standard YOLO inference on full frame."""
        results = self.yolo_model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            half=self._use_half,
            verbose=False,
            device=self.device,
        )
        
        return _extract_yolo_detections(results)

    def _run_sahi(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """SAHI sliced inference."""
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        if self._sahi_model is None:
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=self.model_path,
                confidence_threshold=self.conf_threshold,
                device=self.device,
            )
            
        result = get_sliced_prediction(
            frame,
            self._sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0,
        )
        
        detections = []
        
        for obj in result.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            confidence = obj.score.value
            detections.append((float(x1), float(y1), float(x2), float(y2), float(confidence)))
            
        return detections

    def _run_on_rois(self, frame: np.ndarray, rois: list[tuple[int, int, int, int]]) -> list[tuple[float, float, float, float, float]]:
        """Run batched YOLO on ROI crops and map detections back to full frame."""
        h, w = frame.shape[:2]
        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []

        for rx, ry, rw, rh in rois:
            x2 = min(rx + rw, w)
            y2 = min(ry + rh, h)
            
            crop = frame[ry:y2, rx:x2]
            
            if crop.size == 0:
                continue
            
            crops.append(crop)
            offsets.append((rx, ry))

        if not crops:
            return []

        results = self.yolo_model.predict(
            source=crops,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            half=self._use_half,
            verbose=False,
            device=self.device,
        )

        all_detections = []
        
        for (rx, ry), r in zip(offsets, results):
            for detection in _extract_yolo_detections([r]):
                x1, y1, x2_d, y2_d, conf = detection
                
                all_detections.append((
                    float(x1) + rx,
                    float(y1) + ry,
                    float(x2_d) + rx,
                    float(y2_d) + ry,
                    conf,
                ))
                
        return all_detections

    def _should_use_sahi(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        
        return self.use_sahi or w > self.sahi_threshold or h > self.sahi_threshold

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Run detection on frame. Returns [(x1, y1, x2, y2, confidence), ...]."""
        if frame is None or frame.size == 0:
            return []

        if self.use_motion_roi:
            rois = self._get_motion_rois(frame)
            rois = self._merge_overlapping_rois(rois)
            
            if rois:
                self._motion_roi_fallback_counter = 0
                
                return self._run_on_rois(frame, rois)
            
            self._motion_roi_fallback_counter += 1
            
            # Fall back to full-frame / SAHI every 5th frame when no motion ROIs
            if self._motion_roi_fallback_counter % 5 != 1:
                return []
            

        if self._should_use_sahi(frame):
            return self._run_sahi(frame)
        
        return self._run_full_frame_yolo(frame)

    def calculate_center(self, detections) -> list[tuple[float, float, float]]:
        """Calculate center (u, v) and confidence per detection."""
        return [
            ((x1 + x2) / 2, (y1 + y2) / 2, confidence)
            for x1, y1, x2, y2, confidence in detections
        ]
