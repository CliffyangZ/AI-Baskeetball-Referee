"""
Matching utilities for ByteTrack algorithm
Includes Hungarian algorithm and IoU calculation functions
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

# Use generic types to avoid circular imports
from typing import TypeVar, Any, Protocol

class DetectionProtocol(Protocol):
    bbox: Tuple[float, float, float, float]
    confidence: float

T = TypeVar('T', bound=DetectionProtocol)

def apply_nms(detections: List[T], iou_threshold: float = 0.5) -> List[T]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply NMS
        keep_detections = []
        while detections:
            # Keep the detection with highest confidence
            best_detection = detections.pop(0)
            keep_detections.append(best_detection)
            
            # Remove detections with high IoU overlap
            remaining_detections = []
            for detection in detections:
                iou = calculate_iou(best_detection.bbox, detection.bbox)
                if iou <= iou_threshold:
                    remaining_detections.append(detection)
            
            detections = remaining_detections
        
        return keep_detections

def calculate_iou(bbox1: Tuple[float, float, float, float], 
                 bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate IoU between two bounding boxes
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_iou_matrix(tracks: List, detections: List) -> np.ndarray:
    """
    Calculate IoU matrix between tracks and detections
    
    Args:
        tracks: List of track objects with bbox attribute
        detections: List of detection objects with bbox attribute
        
    Returns:
        IoU matrix of shape (len(tracks), len(detections))
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    iou_matrix = np.zeros((len(tracks), len(detections)))
    for t, track in enumerate(tracks):
        for d, detection in enumerate(detections):
            iou_matrix[t, d] = calculate_iou(track.bbox, detection.bbox)
    
    return iou_matrix


def hungarian_matching(cost_matrix: np.ndarray, 
                      max_cost: float = 0.7) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian algorithm for optimal assignment
    
    Args:
        cost_matrix: Cost matrix for assignment
        max_cost: Maximum cost threshold for valid matches
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    # Convert IoU to cost (1 - IoU)
    cost_matrix = 1 - cost_matrix
    
    # Apply Hungarian algorithm
    track_indices, det_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_detections = list(range(cost_matrix.shape[1]))
    
    # Filter matches by cost threshold
    for t, d in zip(track_indices, det_indices):
        if cost_matrix[t, d] <= (1 - max_cost):  # Convert back to IoU threshold
            matches.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)
    
    return matches, unmatched_tracks, unmatched_detections


def greedy_matching(cost_matrix: np.ndarray, 
                   threshold: float = 0.3) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy matching algorithm as fallback
    
    Args:
        cost_matrix: IoU matrix for matching
        threshold: Minimum IoU threshold for valid matches
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_detections = list(range(cost_matrix.shape[1]))
    
    # Sort potential matches by IoU score (highest first)
    potential_matches = []
    for t in range(cost_matrix.shape[0]):
        for d in range(cost_matrix.shape[1]):
            if cost_matrix[t, d] >= threshold:
                potential_matches.append((cost_matrix[t, d], t, d))
    
    # Sort by IoU score descending
    potential_matches.sort(reverse=True)
    
    # Assign matches greedily
    for iou_score, t, d in potential_matches:
        if t in unmatched_tracks and d in unmatched_detections:
            matches.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)
    
    return matches, unmatched_tracks, unmatched_detections
