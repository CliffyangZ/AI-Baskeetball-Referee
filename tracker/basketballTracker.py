"""
BasketballTracker: YOLO detection + ByteTrack-style two-stage association (see byteTrack_meth.md).

Input/Output spec
-----------------
class BasketballTracker:
    __init__(model_path: Optional[str] = None,
             config_path: Optional[str] = None,
             device: str = 'cpu')
        - Loads Ultralytics YOLO model. If model_path is None, uses 'yolov8n.pt'.
        - Loads ByteTrack-like thresholds from YAML. If config_path is None, loads
          '<this_dir>/basketball_bytetrack.yaml'.

    infer_frame(frame: np.ndarray, conf: Optional[float] = None) -> tuple[np.ndarray, dict]
        - frame: BGR image (H,W,3)
        - conf: optional detection threshold override (fallback to YAML high threshold)
        - returns (annotated_frame, meta)
          meta = {
            'conf': float,                   # detection threshold used
            'fps': float,                    # rough processing fps
            'objects': [                     # flat list of detected/tracked objects
               { 'id': Optional[int], 'cls': str, 'bbox': [x1,y1,x2,y2], 'score': float }
            ],
            'tracks': {
               'person': [ { 'id': int, 'bbox': [...], 'score': float } ],
               'sports ball': [ { 'id': int, 'bbox': [...], 'score': float } ]
            }
          }

Notes
-----
- Association: two stages per byteTrack_meth.md
  1) Match high-confidence detections to existing tracks using IoU >= match_thresh
  2) Match remaining tracks with low-confidence detections using IoU >= proximity_thresh
- New tracks are initialized from unmatched high-confidence detections with conf >= new_track_thresh
- Tracks are removed if time_since_update > track_buffer
- This simplified variant uses last bbox (no Kalman) to predict.
"""
from __future__ import annotations

import os
import time
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Track:
    id: int
    bbox: np.ndarray  # [x1,y1,x2,y2]
    score: float
    cls: str
    time_since_update: int = 0
    hits: int = 0

    def update(self, bbox: np.ndarray, score: float):
        self.bbox = bbox
        self.score = score
        self.time_since_update = 0
        self.hits += 1


class BasketballTracker:
    def __init__(
        self,
        model_path: Optional[str] = "pt_models/basketballModel.pt",
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = YOLO(model_path)

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "basketball_bytetrack.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # BYTETrack-like thresholds
        self.match_thresh: float = float(cfg.get("match_thresh", 0.8))
        self.new_track_thresh: float = float(cfg.get("new_track_thresh", 0.6))
        self.proximity_thresh: float = float(cfg.get("proximity_thresh", 0.5))
        self.track_buffer: int = int(cfg.get("track_buffer", 40))
        self.track_high_thresh: float = float(cfg.get("track_high_thresh", 0.5))
        self.track_low_thresh: float = float(cfg.get("track_low_thresh", 0.1))

        # trackers per class
        self.next_id = 1
        self.tracks: Dict[str, List[Track]] = {
            "person": [],
            "sports ball": [],
        }
        self.class_map = {0: "person", 32: "sports ball"}  # COCO ids commonly
        self.t_last = time.time()

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

    @staticmethod
    def _centroid(b: np.ndarray) -> Tuple[float, float]:
        return float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)

    def _assign(self, tracks: List[Track], dets: np.ndarray, iou_thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU matching to avoid SciPy dependency.
        Returns (matches, unmatched_track_indices, unmatched_det_indices).
        """
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        T = len(tracks)
        D = len(dets)
        ious = np.zeros((T, D), dtype=np.float32)
        for i, tr in enumerate(tracks):
            for j in range(D):
                ious[i, j] = self._iou(tr.bbox, dets[j, :4])
        matched_t = set()
        matched_d = set()
        matches: List[Tuple[int, int]] = []
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
            matches.append((i, j))
            matched_t.add(i)
            matched_d.add(j)
        unmatched_t = [i for i in range(T) if i not in matched_t]
        unmatched_d = [j for j in range(D) if j not in matched_d]
        return matches, unmatched_t, unmatched_d

    def _update_class_tracks(self, cls_name: str, high_dets: np.ndarray, low_dets: np.ndarray) -> None:
        # Stage 1: match high-confidence dets
        tracks = self.tracks[cls_name]
        matches, unmatched_t, unmatched_d = self._assign(tracks, high_dets, self.match_thresh)
        for ti, dj in matches:
            bbox = high_dets[dj, :4]
            score = float(high_dets[dj, 4])
            tracks[ti].update(bbox, score)
        # Stage 2: match remaining tracks with low-confidence dets
        if len(unmatched_t) > 0 and len(low_dets) > 0:
            rem_tracks = [tracks[i] for i in unmatched_t]
            m2, rem_t_idx, rem_d_idx = self._assign(rem_tracks, low_dets, self.proximity_thresh)
            # map back track indices
            for (rt_i, dj) in m2:
                ti = unmatched_t[rt_i]
                bbox = low_dets[dj, :4]
                score = float(low_dets[dj, 4])
                tracks[ti].update(bbox, score)
                unmatched_t.remove(ti)
        # Create new tracks from unmatched high_dets above new_track_thresh
        for dj in unmatched_d:
            conf = float(high_dets[dj, 4])
            if conf >= self.new_track_thresh:
                bbox = high_dets[dj, :4]
                tr = Track(id=self.next_id, bbox=bbox, score=conf, cls=cls_name)
                self.next_id += 1
                tracks.append(tr)
        # Age unmatched tracks and prune
        still = []
        for idx, tr in enumerate(tracks):
            if all(idx != ti for ti, _ in matches):
                tr.time_since_update += 1
            if tr.time_since_update <= self.track_buffer:
                still.append(tr)
        self.tracks[cls_name] = still

    def _split_by_conf(self, dets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if dets.size == 0:
            return dets, dets
        confs = dets[:, 4]
        high = dets[confs >= self.track_high_thresh]
        low = dets[(confs < self.track_high_thresh) & (confs >= self.track_low_thresh)]
        return high, low

    def infer_frame(self, frame: np.ndarray, conf: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        t0 = time.time()
        det_conf = float(conf) if conf is not None else float(self.track_low_thresh)
        results = self.model.predict(source=frame, conf=det_conf, verbose=False, device=self.device)
        r = results[0]
        boxes = r.boxes
        det_xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.zeros((0, 4), dtype=np.float32)
        det_conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.zeros((0,), dtype=np.float32)
        det_cls_arr = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.zeros((0,), dtype=int)

        # Split detections by class (person, sports ball)
        per_mask = det_cls_arr == 0
        ball_mask = det_cls_arr == 32
        per_dets = np.hstack([det_xyxy[per_mask], det_conf_arr[per_mask, None]]) if det_xyxy.size else np.zeros((0,5), dtype=np.float32)
        ball_dets = np.hstack([det_xyxy[ball_mask], det_conf_arr[ball_mask, None]]) if det_xyxy.size else np.zeros((0,5), dtype=np.float32)

        per_high, per_low = self._split_by_conf(per_dets)
        ball_high, ball_low = self._split_by_conf(ball_dets)

        # Update trackers
        self._update_class_tracks("person", per_high, per_low)
        self._update_class_tracks("sports ball", ball_high, ball_low)

        # Build annotation
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for cls_name, tracks in self.tracks.items():
            color = (0, 255, 0) if cls_name == "person" else (0, 165, 255)
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name[:6]}#{tr.id}", (x1, max(0, y1-6)), font, 0.5, color, 1, cv2.LINE_AA)

        # Build meta
        meta_objects = []
        out_tracks = {"person": [], "sports ball": []}
        for cls_name, tracks in self.tracks.items():
            for tr in tracks:
                o = {
                    "id": int(tr.id),
                    "cls": cls_name,
                    "bbox": [float(v) for v in tr.bbox.tolist()],
                    "score": float(tr.score),
                }
                meta_objects.append(o)
                out_tracks[cls_name].append({"id": int(tr.id), "bbox": o["bbox"], "score": o["score"]})

        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - self.t_last))
        self.t_last = t1
        meta = {
            "conf": det_conf,
            "fps": float(fps),
            "objects": meta_objects,
            "tracks": out_tracks,
        }
        return annotated, meta


__all__ = ["BasketballTracker", "Track"]
