"""
Drawing utilities for basketball tracking visualization
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_bbox(frame: np.ndarray, 
              bbox: Tuple[float, float, float, float],
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on frame
    
    Args:
        frame: Input frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_center_point(frame: np.ndarray,
                     center: Tuple[float, float],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     radius: int = 5) -> np.ndarray:
    """
    Draw center point on frame
    
    Args:
        frame: Input frame
        center: Center coordinates (x, y)
        color: BGR color tuple
        radius: Circle radius
        
    Returns:
        Frame with drawn center point
    """
    center_x, center_y = map(int, center)
    cv2.circle(frame, (center_x, center_y), radius, color, -1)
    return frame


def draw_trajectory(frame: np.ndarray,
                   trajectory: List[Tuple[float, float]],
                   color: Tuple[int, int, int] = (255, 0, 0),
                   thickness: int = 2) -> np.ndarray:
    """
    Draw trajectory line on frame
    
    Args:
        frame: Input frame
        trajectory: List of trajectory points
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with drawn trajectory
    """
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            pt1 = tuple(map(int, trajectory[i-1]))
            pt2 = tuple(map(int, trajectory[i]))
            cv2.line(frame, pt1, pt2, color, thickness)
    return frame


def draw_text_label(frame: np.ndarray,
                   text: str,
                   position: Tuple[int, int],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   font_scale: float = 0.6,
                   thickness: int = 2,
                   background: bool = True) -> np.ndarray:
    """
    Draw text label on frame
    
    Args:
        frame: Input frame
        text: Text to draw
        position: Text position (x, y)
        color: Text color
        font_scale: Font scale
        thickness: Text thickness
        background: Whether to draw background rectangle
        
    Returns:
        Frame with drawn text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if background:
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x, y - text_height - baseline),
                     (x + text_width, y + baseline),
                     (0, 0, 0), -1)
    
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    return frame


def draw_basketball_track(frame: np.ndarray,
                         track,
                         show_trajectory: bool = True,
                         show_velocity: bool = True) -> np.ndarray:
    """
    Draw complete basketball track visualization
    
    Args:
        frame: Input frame
        track: Basketball track object
        show_trajectory: Whether to show trajectory
        show_velocity: Whether to show velocity info
        
    Returns:
        Frame with complete track visualization
    """
    # Draw bounding box
    frame = draw_bbox(frame, track.bbox, (0, 255, 0), 2)
    
    # Draw center point
    frame = draw_center_point(frame, track.center, (0, 255, 0), 5)
    
    # Draw trajectory
    if show_trajectory and hasattr(track, 'trajectory') and len(track.trajectory) > 1:
        frame = draw_trajectory(frame, track.trajectory, (255, 0, 0), 2)
    
    # Draw track ID and confidence
    x1, y1, x2, y2 = map(int, track.bbox)
    label = f"Ball {track.track_id}: {track.confidence:.2f}"
    frame = draw_text_label(frame, label, (x1, y1 - 10), (0, 255, 0))
    
    # Draw velocity info
    if show_velocity and hasattr(track, 'velocity'):
        vx, vy = track.velocity
        vel_label = f"V: ({vx:.1f}, {vy:.1f})"
        frame = draw_text_label(frame, vel_label, (x1, y2 + 20), (255, 0, 0), 0.4)
    
    return frame


def draw_detection_info(frame: np.ndarray,
                       detections: List,
                       tracks: List,
                       fps: Optional[float] = None) -> np.ndarray:
    """
    Draw detection statistics on frame
    
    Args:
        frame: Input frame
        detections: List of detections
        tracks: List of active tracks
        fps: Current FPS
        
    Returns:
        Frame with detection info
    """
    height, width = frame.shape[:2]
    
    # Draw detection count
    det_text = f"Detections: {len(detections)}"
    frame = draw_text_label(frame, det_text, (10, 30), (255, 255, 255))
    
    # Draw track count
    track_text = f"Active Tracks: {len(tracks)}"
    frame = draw_text_label(frame, track_text, (10, 60), (255, 255, 255))
    
    # Draw FPS if available
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        frame = draw_text_label(frame, fps_text, (10, 90), (255, 255, 255))
    
    return frame
