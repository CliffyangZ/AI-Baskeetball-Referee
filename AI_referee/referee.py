"""
AIReferee: Simple rules-based foul/event detector using BasketballTracker and PoseTracker.

Input/Output spec
-----------------
class AIReferee:
    __init__(
        bb_tracker: Optional[BasketballTracker] = None,
        pose_tracker: Optional[PoseTracker] = None,
        thresholds: Optional[dict] = None,
    ) -> None
        - bb_tracker: existing BasketballTracker instance or created internally with defaults.
        - pose_tracker: existing PoseTracker instance or created internally with defaults.
        - thresholds: dict overrides for:
            {
              'contact_iou': 0.05,              # IoU between two person boxes to consider contact
              'possession_dist_px': 60.0,       # max centroid distance person<->ball for possession
              'hand_inside_body_margin_px': 12, # margin for checking wrist inside body bbox
              'reach_in_min_frames': 2,         # require N consecutive frames to emit reach_in
            }

    infer_frame(
        frame: np.ndarray,
        bb_conf: Optional[float] = None,
        pose_conf: float = 0.25,
    ) -> tuple[np.ndarray, dict]
        - frame: BGR image (H,W,3)
        - bb_conf: optional detection threshold override passed to BasketballTracker
        - pose_conf: confidence threshold for PoseTracker
        - returns (annotated_frame, meta)
          meta = {
            'events': [ # list of event dicts
               { 'type': 'possession', 'player_id': int, 'ball_bbox': [x1,y1,x2,y2], 'score': float },
               { 'type': 'contact', 'p1': int, 'p2': int, 'iou': float },
               { 'type': 'reach_in', 'defender_id': int, 'offender_id': int, 'hand': 'left'|'right', 'confidence': float },
            ],
            'possession': { 'player_id': Optional[int], 'ball_bbox': Optional[list[float]] },
            'bb_meta': <BasketballTracker meta>,
            'pose_meta': <PoseTracker meta>,
            'frame_index': int,
          }

Notes
-----
- BasketballTracker follows a ByteTrack-inspired two-stage association (see byteTrack_meth.md) to provide stable IDs.
- This module applies simple geometric heuristics, intended as a baseline; refine rules as needed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import cv2
import numpy as np

try:
    from tracker import BasketballTracker, PoseTracker
except Exception:
    # Fallback for relative import if used as a package-less script
    from ..tracker import BasketballTracker, PoseTracker  # type: ignore


class AIReferee:
    def __init__(
        self,
        bb_tracker: Optional[BasketballTracker] = None,
        pose_tracker: Optional[PoseTracker] = None,
        thresholds: Optional[dict] = None,
    ) -> None:
        self.bb_tracker = bb_tracker or BasketballTracker()
        self.pose_tracker = pose_tracker or PoseTracker()
        th = thresholds or {}
        self.contact_iou: float = float(th.get('contact_iou', 0.05))
        self.possession_dist_px: float = float(th.get('possession_dist_px', 60.0))
        self.hand_inside_body_margin_px: float = float(th.get('hand_inside_body_margin_px', 12.0))
        self.reach_in_min_frames: int = int(th.get('reach_in_min_frames', 2))

        self.last_possessor_id: Optional[int] = None
        self.frame_index: int = 0
        # Track consecutive frames of suspected reach-in per defender->offender mapping
        self._reach_in_counters: Dict[Tuple[int, int, str], int] = {}

    @staticmethod
    def _centroid(b: List[float]) -> Tuple[float, float]:
        return (float(b[0] + b[2]) * 0.5, float(b[1] + b[3]) * 0.5)

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        a2 = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return float(inter / (a1 + a2 - inter + 1e-6))

    @staticmethod
    def _inflate(b: List[float], m: float) -> List[float]:
        return [b[0]-m, b[1]-m, b[2]+m, b[3]+m]

    @staticmethod
    def _point_in_bbox(pt: Tuple[float, float], b: List[float]) -> bool:
        return (b[0] <= pt[0] <= b[2]) and (b[1] <= pt[1] <= b[3])

    def _find_possessor(self, persons: List[Dict], balls: List[Dict]) -> Optional[Tuple[int, List[float]]]:
        if not balls:
            return None
        # pick nearest person to the main ball (assume one ball: choose highest score)
        ball = max(balls, key=lambda o: o['score'])
        bc = self._centroid(ball['bbox'])
        best = (None, 1e9)
        for p in persons:
            pb = p['bbox']
            pc = self._centroid(pb)
            d = math.hypot(pc[0]-bc[0], pc[1]-bc[1])
            if self._point_in_bbox(bc, self._inflate(pb, 6.0)) or d < best[1]:
                best = (p['id'], d)
        pid, dist = best
        if pid is not None and dist <= self.possession_dist_px:
            return int(pid), ball['bbox']
        return None

    def _detect_contacts(self, persons: List[Dict]) -> List[Dict]:
        events = []
        for i in range(len(persons)):
            for j in range(i+1, len(persons)):
                a, b = persons[i], persons[j]
                iou = self._iou(a['bbox'], b['bbox'])
                if iou >= self.contact_iou:
                    events.append({'type': 'contact', 'p1': int(a['id']), 'p2': int(b['id']), 'iou': float(iou)})
        return events

    def _detect_reach_in(self, possessor_id: Optional[int], persons: List[Dict], people_kps: List[Dict], ball_bbox: Optional[List[float]]) -> List[Dict]:
        events = []
        if possessor_id is None or ball_bbox is None:
            # reset counters if no possession
            self._reach_in_counters.clear()
            return events
        # locate offender (possessor) body bbox
        offender_bbox = None
        for p in persons:
            if p['id'] == possessor_id:
                offender_bbox = p['bbox']
                break
        if offender_bbox is None:
            return events
        # For each other person with keypoints, check wrists inside offender bbox near the ball
        inflated = self._inflate(offender_bbox, self.hand_inside_body_margin_px)
        ball_c = self._centroid(ball_bbox)
        for pers in people_kps:
            pid = pers.get('id')
            if pid is None or pid == possessor_id:
                continue
            # gather wrists
            left = next((k for k in pers['keypoints'] if k['name'] == 'left_wrist'), None)
            right = next((k for k in pers['keypoints'] if k['name'] == 'right_wrist'), None)
            for hand_name, kp in [('left', left), ('right', right)]:
                if kp is None:
                    continue
                if kp['conf'] < 0.3:
                    continue
                pt = (kp['x'], kp['y'])
                # check entering offender bbox and closeness to ball
                if self._point_in_bbox(pt, inflated):
                    d_ball = math.hypot(pt[0]-ball_c[0], pt[1]-ball_c[1])
                    key = (int(pid), int(possessor_id), hand_name)
                    if d_ball < self.possession_dist_px * 0.8:  # tighter near-ball criterion
                        self._reach_in_counters[key] = self._reach_in_counters.get(key, 0) + 1
                        if self._reach_in_counters[key] >= self.reach_in_min_frames:
                            events.append({
                                'type': 'reach_in',
                                'defender_id': int(pid),
                                'offender_id': int(possessor_id),
                                'hand': hand_name,
                                'confidence': float(min(1.0, 0.5 + 0.5 * (self._reach_in_counters[key] / max(1, self.reach_in_min_frames))))
                            })
                    else:
                        # not close to ball: decay
                        self._reach_in_counters[key] = max(0, self._reach_in_counters.get(key, 0) - 1)
                else:
                    # outside offender bbox: decay
                    self._reach_in_counters[(int(pid), int(possessor_id), hand_name)] = max(0, self._reach_in_counters.get((int(pid), int(possessor_id), hand_name), 0) - 1)
        return events

    def infer_frame(self, frame: np.ndarray, bb_conf: Optional[float] = None, pose_conf: float = 0.25) -> Tuple[np.ndarray, dict]:
        self.frame_index += 1
        # Step 1: bounding-box tracker (YOLO + ByteTrack-like association)
        bb_annot, bb_meta = self.bb_tracker.infer_frame(frame, conf=bb_conf)
        persons = bb_meta['tracks'].get('person', [])
        balls = bb_meta['tracks'].get('sports ball', [])

        # Step 2: pose inference aligned to person IDs
        pose_annot, pose_meta = self.pose_tracker.infer_frame(frame, persons=persons, conf=pose_conf)

        # Events
        events: List[Dict] = []
        possession = {'player_id': None, 'ball_bbox': None}
        poss = self._find_possessor(persons, balls)
        if poss is not None:
            pid, ball_bbox = poss
            possession = {'player_id': pid, 'ball_bbox': ball_bbox}
            if pid != self.last_possessor_id:
                events.append({'type': 'possession', 'player_id': pid, 'ball_bbox': ball_bbox, 'score': 1.0})
            self.last_possessor_id = pid
        else:
            self.last_possessor_id = None

        events.extend(self._detect_contacts(persons))
        events.extend(self._detect_reach_in(possession['player_id'], persons, pose_meta.get('people', []), possession['ball_bbox']))

        # Compose annotation: start with original frame, overlay bb+pose minimal markers
        annotated = frame.copy()
        # draw persons with IDs from bb tracker
        for p in persons:
            x1, y1, x2, y2 = map(int, p['bbox'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"P#{p['id']}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        # draw ball
        for b in balls:
            x1, y1, x2, y2 = map(int, b['bbox'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(annotated, "BALL", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)
        # pose keypoints minimal
        for pers in pose_meta.get('people', []):
            for k in pers.get('keypoints', []):
                if k.get('conf', 0.0) >= 0.3:
                    cv2.circle(annotated, (int(k['x']), int(k['y'])), 2, (0, 255, 255), -1)
        # draw possession link
        if possession['player_id'] is not None and possession['ball_bbox'] is not None:
            pid = possession['player_id']
            pb = next((p for p in persons if p['id'] == pid), None)
            if pb is not None:
                pc = self._centroid(pb['bbox'])
                bc = self._centroid(possession['ball_bbox'])
                cv2.line(annotated, (int(pc[0]), int(pc[1])), (int(bc[0]), int(bc[1])), (255, 0, 0), 2)

        meta = {
            'events': events,
            'possession': possession,
            'bb_meta': bb_meta,
            'pose_meta': pose_meta,
            'frame_index': self.frame_index,
        }
        return annotated, meta


__all__ = ['AIReferee']
