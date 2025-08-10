"""
PoseTracker: YOLOv8 Pose inference with optional alignment to person tracks.

Input/Output spec
-----------------
class PoseTracker:
    __init__(model_path: Optional[str] = None, device: str = 'cpu')
        - Loads Ultralytics YOLO pose model. If model_path is None, uses 'yolov8n-pose.pt'.

    infer_frame(
        frame: np.ndarray,
        persons: Optional[list[dict]] = None,  # from BasketballTracker meta['tracks']['person']
        conf: float = 0.25,
    ) -> tuple[np.ndarray, dict]
        - frame: BGR image (H,W,3)
        - persons: optional list of {id:int, bbox:[x1,y1,x2,y2], score:float}
        - returns (annotated_frame, meta)
          meta = {
            'conf': float,
            'fps': float,
            'people': [
               {
                 'id': Optional[int],              # aligned with provided persons by IoU (if any)
                 'bbox': [x1,y1,x2,y2],
                 'score': float,
                 'keypoints': [
                    { 'name': str, 'x': float, 'y': float, 'conf': float }
                 ]
               }
            ]
          }

Notes
-----
- If persons is provided, we associate pose detections to those bboxes using IoU >= 0.3 and assign their IDs.
- If persons is None, IDs will be None (no internal tracking implemented to keep responsibilities separated).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import time

import os
import cv2
import numpy as np
from ultralytics import YOLO

def _is_openvino_model_path(p: Optional[str]) -> bool:
    """Return True if path points to an OpenVINO IR (.xml) file or a directory containing one."""
    if not p:
        return False
    try:
        if os.path.isdir(p):
            return any(name.lower().endswith(".xml") for name in os.listdir(p))
        return str(p).lower().endswith(".xml")
    except Exception:
        return False


COCO_KPT_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]


class PoseTracker:
    def __init__(self, model_path: Optional[str] = "pt_models/yolov8s-pose.pt", device: str = 'cpu') -> None:
        self.is_openvino = _is_openvino_model_path(model_path)
        model_load_path = model_path
        if self.is_openvino and model_path and os.path.isdir(model_path):
            xmls = [os.path.join(model_path, n) for n in os.listdir(model_path) if n.lower().endswith('.xml')]
            if not xmls:
                raise FileNotFoundError(f"No .xml found in OpenVINO dir: {model_path}")
            model_load_path = xmls[0]
        if self.is_openvino:
            print(f"[PoseTracker] Using OpenVINO IR model: {model_load_path}")
        else:
            print(f"[PoseTracker] Using model: {model_load_path}")
        self.model = YOLO(model_load_path)
        self.device = device
        self._last_time = time.time()

    @staticmethod
    def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter + 1e-6
        return float(inter / union)

    def _assign_to_persons(
        self,
        pose_boxes: np.ndarray,  # Nx4
        persons: List[Dict],
        iou_thresh: float = 0.3,
    ) -> Tuple[List[Optional[int]], List[int]]:
        """Greedy IoU matching pose boxes to provided person tracks (no SciPy).
        Returns (assigned_ids_per_pose_box, unmatched_pose_indices).
        """
        D = len(pose_boxes)
        if len(persons) == 0 or D == 0:
            return [None] * D, list(range(D))
        T = len(persons)
        ious = np.zeros((T, D), dtype=np.float32)
        for i, p in enumerate(persons):
            pb = np.array(p['bbox'], dtype=np.float32)
            for j in range(D):
                ious[i, j] = self._iou(pb, pose_boxes[j])
        matched_t = set()
        matched_d = set()
        assigned_ids: List[Optional[int]] = [None] * D
        while True:
            best_iou = -1.0
            best = (-1, -1)
            for i in range(T):
                if i in matched_t:
                    continue
                for j in range(D):
                    if j in matched_d:
                        continue
                    v = float(ious[i, j])
                    if v > best_iou:
                        best_iou = v
                        best = (i, j)
            if best_iou < iou_thresh or best[0] == -1:
                break
            i, j = best
            assigned_ids[j] = int(persons[i]['id']) if 'id' in persons[i] else None
            matched_t.add(i)
            matched_d.add(j)
        unmatched_d = [j for j in range(D) if j not in matched_d]
        return assigned_ids, unmatched_d

    def infer_frame(
        self,
        frame: np.ndarray,
        persons: Optional[List[Dict]] = None,
        conf: float = 0.25,
    ) -> Tuple[np.ndarray, dict]:
        t0 = time.time()
        results = self.model.predict(source=frame, conf=conf, verbose=False, device=self.device)
        r = results[0]

        boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r, 'boxes') and hasattr(r.boxes, 'xyxy') else np.zeros((0, 4), dtype=np.float32)
        boxes_conf = r.boxes.conf.cpu().numpy() if hasattr(r, 'boxes') and hasattr(r.boxes, 'conf') else np.zeros((0,), dtype=np.float32)

        kp_xy = None
        kp_conf = None
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            # Prefer absolute xy
            if hasattr(r.keypoints, 'xy') and r.keypoints.xy is not None:
                kp_xy = r.keypoints.xy.cpu().numpy()  # (N, K, 2)
            elif hasattr(r.keypoints, 'xyn') and r.keypoints.xyn is not None:
                # Convert normalized to absolute using image size
                xyn = r.keypoints.xyn.cpu().numpy()
                H, W = frame.shape[:2]
                kp_xy = xyn.copy()
                kp_xy[..., 0] *= W
                kp_xy[..., 1] *= H
            if hasattr(r.keypoints, 'conf') and r.keypoints.conf is not None:
                try:
                    kp_conf = r.keypoints.conf.cpu().numpy()  # (N, K)
                except Exception:
                    kp_conf = None
        else:
            kp_xy = np.zeros((0, len(COCO_KPT_NAMES), 2), dtype=np.float32)

        N = boxes_xyxy.shape[0]
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        people_out: List[Dict] = []

        assigned_ids: List[Optional[int]] = [None] * N
        if persons is not None:
            assigned_ids, _ = self._assign_to_persons(boxes_xyxy, persons, iou_thresh=0.3)

        for i in range(N):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            score = float(boxes_conf[i]) if i < len(boxes_conf) else 0.0
            pid = assigned_ids[i]

            # draw bbox
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"P#{pid}" if pid is not None else f"P?"
            cv2.putText(annotated, label, (x1, max(0, y1 - 6)), font, 0.5, color, 1, cv2.LINE_AA)

            # draw keypoints
            person_kps = []
            if kp_xy is not None and i < kp_xy.shape[0]:
                for k, name in enumerate(COCO_KPT_NAMES):
                    x, y = float(kp_xy[i, k, 0]), float(kp_xy[i, k, 1])
                    c = float(kp_conf[i, k]) if (kp_conf is not None and i < kp_conf.shape[0]) else 1.0
                    person_kps.append({'name': name, 'x': x, 'y': y, 'conf': c})
                    if c > 0.3:
                        cv2.circle(annotated, (int(x), int(y)), 2, (0, 255, 255), -1)

            people_out.append({
                'id': pid,
                'bbox': [float(v) for v in boxes_xyxy[i].tolist()],
                'score': score,
                'keypoints': person_kps,
            })

        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - self._last_time))
        self._last_time = t1
        meta = {
            'conf': float(conf),
            'fps': float(fps),
            'people': people_out,
        }
        return annotated, meta


__all__ = ['PoseTracker', 'COCO_KPT_NAMES']
