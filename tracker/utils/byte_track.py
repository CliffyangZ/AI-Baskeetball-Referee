"""
ByteTrack implementation for basketball tracking
Based on Ultralytics ByteTracker with basketball-specific optimizations
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .KalmanFilter import BasketballKalmanFilter
from .matching import calculate_iou_matrix, hungarian_matching, greedy_matching

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track state enumeration"""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class BasketballDetection:
    """Basketball detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float] = (0.0, 0.0)
    class_id: int = 0
    
    def __post_init__(self):
        """Calculate center from bounding box"""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)


class STrack:
    """
    Single track class for ByteTrack algorithm
    Based on Ultralytics STrack implementation
    """
    
    shared_kalman = BasketballKalmanFilter()
    track_id_count = 0
    
    def __init__(self, detection: BasketballDetection, track_id: Optional[int] = None):
        """Initialize track with detection"""
        self.detection = detection
        self.bbox = detection.bbox
        self.center = detection.center
        self.confidence = detection.confidence
        
        # Track management
        self.track_id = track_id if track_id is not None else self._next_id()
        self.state = TrackState.NEW
        self.is_activated = False
        
        # Tracking history
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        
        # Motion prediction
        self.kalman_filter = BasketballKalmanFilter()
        self.mean = None
        self.covariance = None
        
        # Basketball specific
        self.trajectory = []
        self.velocity = (0.0, 0.0)
        
    @classmethod
    def _next_id(cls):
        """Get next track ID"""
        cls.track_id_count += 1
        return cls.track_id_count
    
    @classmethod
    def reset_id(cls):
        """Reset track ID counter"""
        cls.track_id_count = 0
    
    def activate(self, frame_id: int):
        """Activate track"""
        self.track_id = self._next_id()
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        # Initialize Kalman filter
        if self.kalman_filter:
            cx, cy = self.center
            self.kalman_filter.initialize(np.array([cx, cy, 0, 0]))
    
    def re_activate(self, detection: BasketballDetection, frame_id: int):
        """Re-activate lost track"""
        self.update(detection, frame_id)
        self.state = TrackState.TRACKED
        self.is_activated = True
        
    def update(self, detection: BasketballDetection, frame_id: int):
        """Update track with new detection"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        # Calculate velocity
        if len(self.trajectory) > 0:
            prev_center = self.trajectory[-1]
            curr_center = detection.center
            self.velocity = (
                curr_center[0] - prev_center[0],
                curr_center[1] - prev_center[1]
            )
        
        # Update detection info
        self.detection = detection
        self.bbox = detection.bbox
        self.center = detection.center
        self.confidence = detection.confidence
        
        # Update trajectory
        self.trajectory.append(detection.center)
        if len(self.trajectory) > 50:  # Keep last 50 points
            self.trajectory.pop(0)
        
        # Update Kalman filter
        if self.kalman_filter and self.kalman_filter.initialized:
            try:
                cx, cy = detection.center
                self.kalman_filter.update(np.array([cx, cy]))
            except:
                pass
        
        self.state = TrackState.TRACKED
    
    def predict(self):
        """Predict next position using Kalman filter"""
        if self.kalman_filter and self.kalman_filter.initialized:
            try:
                predicted_pos = self.kalman_filter.predict()
                if predicted_pos is not None and len(predicted_pos) >= 4:
                    cx, cy, vx, vy = predicted_pos[:4]
                    # Update predicted position
                    w = self.bbox[2] - self.bbox[0]
                    h = self.bbox[3] - self.bbox[1]
                    self.bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
                    self.center = (cx, cy)
                    self.velocity = (vx, vy)
            except:
                pass
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = TrackState.LOST
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = TrackState.REMOVED


class BYTETracker:
    """
    ByteTrack tracker for basketball detection
    Implements the ByteTrack algorithm with basketball-specific optimizations
    """
    
    def __init__(self, 
                 high_thresh: float = 0.6,
                 low_thresh: float = 0.1,
                 match_thresh: float = 0.8,
                 track_buffer: int = 30,
                 frame_rate: int = 30):
        """
        Initialize ByteTracker
        
        Args:
            high_thresh: High confidence threshold for first association
            low_thresh: Low confidence threshold for second association
            match_thresh: IoU threshold for track association
            track_buffer: Number of frames to keep lost tracks
            frame_rate: Video frame rate
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        # Track management
        self.tracked_stracks = []  # Active tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        logger.info(f"BYTETracker initialized: high_thresh={high_thresh}, low_thresh={low_thresh}")
    
    def update(self, detections: List[BasketballDetection]) -> List[STrack]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of basketball detections
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # Separate detections by confidence
        high_dets = [det for det in detections if det.confidence >= self.high_thresh]
        low_dets = [det for det in detections if self.low_thresh <= det.confidence < self.high_thresh]
        
        # Predict current positions of existing tracks
        for track in self.tracked_stracks + self.lost_stracks:
            track.predict()
        
        # First association: high confidence detections with tracked tracks
        if len(high_dets) > 0:
            # Calculate IoU matrix
            iou_matrix = calculate_iou_matrix(self.tracked_stracks, high_dets)
            
            # Hungarian matching
            matches, unmatched_tracks, unmatched_dets = hungarian_matching(
                iou_matrix, self.match_thresh
            )
            
            # Update matched tracks
            for track_idx, det_idx in matches:
                track = self.tracked_stracks[track_idx]
                detection = high_dets[det_idx]
                track.update(detection, self.frame_id)
                activated_stracks.append(track)
            
            # Handle unmatched tracks from first association
            for track_idx in unmatched_tracks:
                track = self.tracked_stracks[track_idx]
                if track.state != TrackState.LOST:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            # No high confidence detections, mark all tracked as lost
            unmatched_dets = []
            for track in self.tracked_stracks:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Second association: low confidence detections with lost tracks
        if len(low_dets) > 0 and len(lost_stracks) > 0:
            iou_matrix = calculate_iou_matrix(lost_stracks, low_dets)
            matches, unmatched_lost, unmatched_low_dets = hungarian_matching(
                iou_matrix, 0.5  # Lower threshold for lost tracks
            )
            
            # Re-activate matched lost tracks
            for track_idx, det_idx in matches:
                track = lost_stracks[track_idx]
                detection = low_dets[det_idx]
                track.re_activate(detection, self.frame_id)
                refind_stracks.append(track)
            
            # Update remaining lost tracks
            for track_idx in unmatched_lost:
                track = lost_stracks[track_idx]
                if self.frame_id - track.frame_id <= self.max_time_lost:
                    lost_stracks.append(track)
                else:
                    track.mark_removed()
                    removed_stracks.append(track)
        
        # Third association: remaining high confidence detections with lost tracks
        if len(unmatched_dets) > 0 and len(self.lost_stracks) > 0:
            remaining_high_dets = [high_dets[i] for i in unmatched_dets]
            iou_matrix = calculate_iou_matrix(self.lost_stracks, remaining_high_dets)
            matches, _, _ = hungarian_matching(iou_matrix, 0.7)
            
            for track_idx, det_idx in matches:
                track = self.lost_stracks[track_idx]
                detection = remaining_high_dets[det_idx]
                track.re_activate(detection, self.frame_id)
                refind_stracks.append(track)
                unmatched_dets.remove(det_idx)
        
        # Initialize new tracks for remaining unmatched detections
        for det_idx in unmatched_dets:
            detection = high_dets[det_idx]
            track = STrack(detection)
            track.activate(self.frame_id)
            activated_stracks.append(track)
        
        # Update track lists
        self.tracked_stracks = activated_stracks + refind_stracks
        
        # Remove old lost tracks
        self.lost_stracks = []
        for track in lost_stracks:
            if self.frame_id - track.frame_id <= self.max_time_lost:
                self.lost_stracks.append(track)
            else:
                track.mark_removed()
                removed_stracks.append(track)
        
        # Update removed tracks
        self.removed_stracks.extend(removed_stracks)
        
        return self.tracked_stracks
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()
    
    def get_basketball_coordinates(self) -> List[Dict[str, Any]]:
        """
        Get basketball coordinates for AI referee system
        
        Returns:
            List of dictionaries containing basketball information
        """
        basketball_coords = []
        for track in self.tracked_stracks:
            ball_info = {
                'ball_id': track.track_id,
                'bbox': track.bbox,
                'center': track.center,
                'velocity': track.velocity,
                'confidence': track.confidence,
                'trajectory': track.trajectory.copy(),
                'frame_number': self.frame_id
            }
            basketball_coords.append(ball_info)
        
        return basketball_coords
