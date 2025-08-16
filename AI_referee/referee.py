#!/usr/bin/env python3
"""
Basketball Referee System

Integrates all AI referee components to detect violations and track statistics.
Components integrated:
- Basketball tracking (basketballTracker.py)
- Pose tracking (poseTracker.py)
- Dribble counting (dribble_counting.py)
- Ball holding detection (holding_basketball.py)
- Travel violation detection (travel_detection.py)
- Double dribble detection (double_dribble.py)
- Shot detection and counting (shot_counter.py)
- Step counting (step_counting.py)
"""

import os
import sys
import cv2
import time
import logging
import numpy as np
import asyncio
import threading
import concurrent.futures
import queue
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from pathlib import Path
from functools import lru_cache

# Add project root to path to fix imports
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import trackers
from tracker.basketballTracker import BasketballTracker
from tracker.poseTracker import PoseTracker
from tracker.utils.openvino_utils import DeviceType 

# Import referee components
from AI_referee.dribble_counting import DribbleCounter
from AI_referee.holding_basketball import BallHoldingDetector
from AI_referee.travel_detection import TravelViolationDetector
from AI_referee.double_dribble import DoubleDribbleDetector
from AI_referee.shot_counter import ShotDetector
from AI_referee.step_counting import StepCounter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define component types for task management
class ComponentType(Enum):
    BASKETBALL_TRACKER = "basketball_tracker"
    POSE_TRACKER = "pose_tracker"
    DRIBBLE_COUNTER = "dribble_counter"
    HOLDING_DETECTOR = "holding_detector"
    STEP_COUNTER = "step_counter"
    SHOT_DETECTOR = "shot_detector"
    TRAVEL_DETECTOR = "travel_detector"
    DOUBLE_DRIBBLE_DETECTOR = "double_dribble_detector"

@dataclass
class ProcessingTask:
    """Data class for asynchronous processing tasks"""
    component_type: ComponentType
    frame: np.ndarray
    frame_number: int
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher priority tasks are processed first
    
    def __lt__(self, other):
        """Compare tasks based on priority for priority queue ordering"""
        if not isinstance(other, ProcessingTask):
            return NotImplemented
        # First compare by priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # If priorities are equal, compare by timestamp
        return self.timestamp < other.timestamp
    
    def __eq__(self, other):
        """Check equality based on priority and timestamp only (not frame data)"""
        if not isinstance(other, ProcessingTask):
            return False
        return (self.priority == other.priority and 
                abs(self.timestamp - other.timestamp) < 1e-6)
    
@dataclass
class ProcessingResult:
    """Data class for processing results"""
    component_type: ComponentType
    frame: Optional[np.ndarray] = None
    data: Any = None
    frame_number: int = 0
    processing_time: float = 0.0
    success: bool = True
    error_message: str = ""

@dataclass
class ViolationEvent:
    """Data class for violation events"""
    type: str  # Type of violation (travel, double_dribble, etc.)
    timestamp: float  # Time when violation was detected
    description: str  # Human-readable description
    player_id: Optional[int] = None  # ID of player who committed violation
    confidence: float = 1.0  # Confidence level of detection

@dataclass
class GameStatistics:
    """Data class for game statistics"""
    shot_attempts: int = 0
    shot_makes: int = 0
    dribble_count: int = 0
    step_count: Dict[int, int] = None  # Player ID -> step count
    violations: List[ViolationEvent] = None
    holding_duration: float = 0.0
    fps: float = 0.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.step_count is None:
            self.step_count = {}
        if self.violations is None:
            self.violations = []
    
    def update_dribble_count(self, count: int) -> None:
        """Update the dribble count"""
        self.dribble_count = count
    
    def set_shot_stats(self, makes: int, attempts: int) -> None:
        """Update shot statistics"""
        self.shot_makes = makes
        self.shot_attempts = attempts
    
    def add_violation(self, violation_type: str) -> None:
        """Add a violation event to the statistics"""
        timestamp = time.time()
        violation = ViolationEvent(type=violation_type, timestamp=timestamp)
        self.violations.append(violation)
        
    @property
    def shooting_percentage(self) -> float:
        """Calculate shooting percentage"""
        if self.shot_attempts == 0:
            return 0.0
        return (self.shot_makes / self.shot_attempts) * 100
        
    def get_stats_text(self) -> str:
        """Get formatted statistics text for display"""
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Processing: {self.processing_time*1000:.1f}ms",
            f"Shots: {self.shot_makes}/{self.shot_attempts} ({self.shooting_percentage:.1f}%)",
            f"Dribbles: {self.dribble_count}",
        ]
        
        # Add step counts for each player if available
        if self.step_count:
            for player_id, steps in self.step_count.items():
                stats.append(f"Player {player_id} steps: {steps}")
        
        # Add holding duration if tracking
        if self.holding_duration > 0:
            stats.append(f"Holding: {self.holding_duration:.1f}s")
            
        return "\n".join(stats)
        
    def get_stats(self) -> dict:
        """Get statistics as a dictionary"""
        stats = {
            "fps": self.fps,
            "processing_time": self.processing_time,
            "shot_attempts": self.shot_attempts,
            "shot_makes": self.shot_makes,
            "shooting_percentage": self.shooting_percentage,
            "dribble_count": self.dribble_count,
            "step_count": self.step_count.copy() if self.step_count else {},
            "holding_duration": self.holding_duration,
            "violations": [v.__dict__ for v in self.violations] if self.violations else []
        }
        return stats


# Singleton pattern for model instances
class ModelManager:
    """Singleton manager for AI models to prevent redundant initialization"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._basketball_tracker = None
        self._pose_tracker = None
        self._basketball_model_path = None
        self._pose_model_path = None
        self._device = None
        self._initialized = True
        logger.info("ModelManager initialized")
    
    def initialize_models(self, basketball_model_path, pose_model_path, device):
        """Initialize models if not already initialized or if paths have changed"""
        with self._lock:
            # Check if we need to reinitialize
            reinit_basketball = (self._basketball_tracker is None or 
                              self._basketball_model_path != basketball_model_path or
                              self._device != device)
                              
            reinit_pose = (self._pose_tracker is None or 
                        self._pose_model_path != pose_model_path or
                        self._device != device)
            
            # Convert device to string if it's an enum
            device_str = device.value if hasattr(device, 'value') else str(device)
            
            # Initialize basketball tracker if needed
            if reinit_basketball:
                try:
                    logger.info(f"Initializing basketball tracker with model {basketball_model_path}")
                    self._basketball_tracker = BasketballTracker(basketball_model_path, device_str)
                    self._basketball_model_path = basketball_model_path
                    self._device = device
                except Exception as e:
                    logger.error(f"Failed to initialize basketball tracker: {e}")
                    raise
            
            # Initialize pose tracker if needed
            if reinit_pose:
                try:
                    logger.info(f"Initializing pose tracker with model {pose_model_path}")
                    self._pose_tracker = PoseTracker(pose_model_path, device_str)
                    self._pose_model_path = pose_model_path
                    self._device = device
                except Exception as e:
                    logger.error(f"Failed to initialize pose tracker: {e}")
                    raise
    
    @property
    def basketball_tracker(self):
        """Get the basketball tracker instance"""
        if self._basketball_tracker is None:
            raise ValueError("Basketball tracker not initialized")
        return self._basketball_tracker
    
    @property
    def pose_tracker(self):
        """Get the pose tracker instance"""
        if self._pose_tracker is None:
            raise ValueError("Pose tracker not initialized")
        return self._pose_tracker


# Task processor for asynchronous processing
class TaskProcessor:
    """Manages asynchronous processing of tasks"""
    
    def __init__(self, max_workers=4):
        """Initialize the task processor"""
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.worker_thread = None
        self.results_cache = {}  # frame_number -> {component_type: result}
        self.max_cache_size = 30  # Keep results for the last 30 frames
        self._lock = threading.Lock()
        
    def start(self):
        """Start the task processor"""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()
        logger.info("Task processor started")
        
    def stop(self):
        """Stop the task processor"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.thread_pool.shutdown(wait=False)
        logger.info("Task processor stopped")
        
    def submit_task(self, task, priority=0):
        """Submit a task for processing"""
        if not self.running:
            logger.warning("Task processor not running, task discarded")
            return
            
        # Lower priority number means higher priority
        self.task_queue.put((priority, task))
        
    def get_result(self, block=False, timeout=None):
        """Get a processing result"""
        try:
            return self.result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def get_cached_result(self, frame_number, component_type):
        """Get a cached result for a specific frame and component"""
        with self._lock:
            if frame_number in self.results_cache:
                return self.results_cache[frame_number].get(component_type)
        return None
        
    def get_results(self, component_types=None, timeout=0.1):
        """Get all available results for specified component types
        
        Args:
            component_types: List of component types to retrieve results for, or None for all
            timeout: Maximum time to wait for results
            
        Returns:
            Dictionary mapping component types to their latest results
        """
        results = {}
        start_time = time.time()
        
        # Make sure component_types is a list if provided
        if component_types is not None and not isinstance(component_types, (list, tuple, set)):
            component_types = [component_types]
        
        # Try to get results until timeout
        while time.time() - start_time < timeout:
            try:
                # Get a result from the queue without blocking
                result = self.result_queue.get(block=False)
                if result is not None:
                    # If no specific component types requested or this type is requested
                    if component_types is None or result.component_type in component_types:
                        results[result.component_type] = result
                        
                    # If we have all requested types, we can stop
                    if component_types is not None and all(t in results for t in component_types):
                        break
            except queue.Empty:
                # No more results in queue
                break
                
        return results
        
    def _process_tasks(self):
        """Process tasks from the queue"""
        while self.running:
            try:
                # Get a task from the queue with a timeout
                try:
                    _, task = self.task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                # Process the task
                start_time = time.time()
                result = None
                error_message = ""
                success = False
                
                try:
                    # Get the appropriate processor for this component type
                    processor = self._get_processor(task.component_type)
                    if processor:
                        # Process the frame
                        result = processor(task.frame)
                        success = True
                    else:
                        error_message = f"No processor found for component type {task.component_type}"
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error processing task {task.component_type}: {e}")
                
                # Create the result with better None handling
                processing_time = time.time() - start_time
                result_frame = None
                if isinstance(result, tuple) and result is not None and len(result) > 0:
                    result_frame = result[0]
                elif result is not None and hasattr(result, '__getitem__'):
                    try:
                        result_frame = result[0]
                    except (IndexError, TypeError):
                        result_frame = None
                
                processing_result = ProcessingResult(
                    component_type=task.component_type,
                    frame=result_frame,
                    data=result,
                    frame_number=task.frame_number,
                    processing_time=processing_time,
                    success=success,
                    error_message=error_message
                )
                
                # Put the result in the result queue
                self.result_queue.put(processing_result)
                
                # Cache the result
                with self._lock:
                    if task.frame_number not in self.results_cache:
                        self.results_cache[task.frame_number] = {}
                    self.results_cache[task.frame_number][task.component_type] = processing_result
                    
                    # Clean up old cache entries
                    if len(self.results_cache) > self.max_cache_size:
                        oldest_frame = min(self.results_cache.keys())
                        del self.results_cache[oldest_frame]
                        
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                
    def _get_processor(self, component_type):
        """Get the processor function for a component type"""
        # This will be implemented in the BasketballReferee class
        return None


class BasketballReferee:
    """
    Basketball Referee System
    
    Integrates all AI referee components to detect violations and track statistics.
    Uses asynchronous processing and shared model instances for optimal performance.
    """
    
    def __init__(self, 
                 basketball_model_path="models/ov_models/basketballModel_openvino_model/basketballModel.xml",
                 pose_model_path="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml",
                 video_path=None,
                 device="AUTO",
                 rules="FIBA",
                 async_processing=True,
                 max_workers=6,
                 visualization_level=2,
                 input_resize_factor=1.0):
        """
        Initialize the basketball referee system with optimized performance settings
        
        Args:
            basketball_model_path: Path to basketball detection model
            pose_model_path: Path to pose detection model
            video_path: Path to video file or camera index (e.g., 0)
            device: Device to run inference on (AUTO, CPU, GPU)
            rules: Basketball rules to follow (FIBA, NBA)
            async_processing: Enable asynchronous processing for non-critical components
            max_workers: Maximum number of worker threads for async processing
            visualization_level: Level of visualization detail (0=minimal, 1=medium, 2=full)
            input_resize_factor: Factor to resize input frames (1.0=original size, 0.5=half size)
        """
        logger.info("Initializing Basketball Referee System with optimized settings")
        
        # Store configuration
        self.rules = rules
        self.async_processing = async_processing
        self.visualization_level = visualization_level
        self.input_resize_factor = input_resize_factor
        
        # Convert string device to DeviceType enum
        if isinstance(device, str):
            device_str = device.upper()
            if device_str == "CPU":
                self.device = DeviceType.CPU
            elif device_str == "GPU" or device_str == "CUDA":
                self.device = DeviceType.GPU
            elif device_str == "NPU":
                self.device = DeviceType.NPU
            else:
                self.device = DeviceType.AUTO
        else:
            # Already a DeviceType enum
            self.device = device
            
        logger.info(f"Using device: {self.device} for inference")
        
        # Initialize video source
        self.video_path = video_path
        # Support camera indices (e.g., 0/1) and keywords (camera/webcam)
        cam_index = None
        if isinstance(video_path, int):
            cam_index = video_path
        elif isinstance(video_path, str):
            vp = video_path.strip().lower()
            if vp.isdigit():
                cam_index = int(vp)
            elif vp in ("cam", "camera", "webcam"):
                cam_index = 0
        try:
            if cam_index is not None:
                logger.info(f"Opening camera index {cam_index}")
                self.cap = cv2.VideoCapture(cam_index)
            else:
                logger.info(f"Opening video file {video_path}")
                self.cap = cv2.VideoCapture(video_path)
        except Exception as e:
            logger.error(f"Failed to create VideoCapture: {e}")
            raise
        if not self.cap or not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize model manager (singleton)
        self.model_manager = ModelManager()
        self.model_manager.initialize_models(basketball_model_path, pose_model_path, self.device)
        
        # Get shared tracker instances from model manager
        self.basketball_tracker = self.model_manager.basketball_tracker
        self.pose_tracker = self.model_manager.pose_tracker
        logger.info("Using shared tracker instances from model manager")
        
        # Initialize referee components with shared trackers
        self.dribble_counter = DribbleCounter(shared_basketball_tracker=self.basketball_tracker, 
                                           video_path=video_path)
        self.holding_detector = BallHoldingDetector(shared_pose_tracker=self.pose_tracker,
                                                 shared_basketball_tracker=self.basketball_tracker,
                                                 video_path=video_path, 
                                                 device=device)
        self.step_counter = StepCounter(shared_pose_tracker=self.pose_tracker,
                                      model_path=pose_model_path, 
                                      video_path=video_path)
        self.shot_detector = ShotDetector(shared_basketball_tracker=self.basketball_tracker)
        # Initialize travel detector with the same rules setting
        self.travel_detector = TravelViolationDetector(rules=self.rules)
        # Initialize double dribble detector
        self.double_dribble_detector = DoubleDribbleDetector()
        
        # Initialize statistics
        self.statistics = GameStatistics()
        self.total_shots = 0
        self.total_makes = 0
        
        # State variables
        self.frame_count = 0
        self.last_violation_time = 0
        self.violation_cooldown = 3.0  # seconds
        self.is_holding_ball = False
        self.processing_times = []
        self.fps_history = []
        self.last_frame_time = 0
        
        # Initialize task processor for asynchronous processing
        self.task_processor = TaskProcessor(max_workers=max_workers)
        if self.async_processing:
            self.task_processor.start()
            logger.info(f"Started asynchronous task processor with {max_workers} workers")
        
        # Component results cache
        self.component_results = {}
        self.basketball_tracks = []
        self.pose_tracks = []
        
        # Frame processing settings
        self.frame_interval = {
            ComponentType.BASKETBALL_TRACKER: 1,  # Process every frame (critical)
            ComponentType.POSE_TRACKER: 1,       # Process every frame (critical)
            ComponentType.DRIBBLE_COUNTER: 2,    # Process every 2nd frame
            ComponentType.HOLDING_DETECTOR: 1,   # Process every frame (critical)
            ComponentType.STEP_COUNTER: 2,       # Process every 2nd frame
            ComponentType.SHOT_DETECTOR: 3,      # Process every 3rd frame
            ComponentType.TRAVEL_DETECTOR: 2,    # Process every 2nd frame
            ComponentType.DOUBLE_DRIBBLE_DETECTOR: 2  # Process every 2nd frame
        }
        
        # Visualization settings
        self.show_stats = False  # Disable statistics display
        self.show_violations = True  # Enable violation display
        self.violation_display_time = 3.0  # seconds
        self.current_violations = []  # List of currently displayed violations
        
        # Create component processors mapping
        self._setup_component_processors()
        
        logger.info("Basketball Referee System initialized with optimized settings")
        
    def _setup_component_processors(self):
        """Setup component processors for task processor"""
        self.component_processors = {
            ComponentType.BASKETBALL_TRACKER: self._process_basketball_tracker,
            ComponentType.POSE_TRACKER: self._process_pose_tracker,
            ComponentType.DRIBBLE_COUNTER: self._process_dribble_counter,
            ComponentType.HOLDING_DETECTOR: self._process_holding_detector,
            ComponentType.STEP_COUNTER: self._process_step_counter,
            ComponentType.SHOT_DETECTOR: self._process_shot_detector,
            ComponentType.TRAVEL_DETECTOR: self._process_travel_detector,
            ComponentType.DOUBLE_DRIBBLE_DETECTOR: self._process_double_dribble_detector
        }
        
        # Override the task processor's _get_processor method
        self.task_processor._get_processor = self._get_component_processor
        
    def _get_component_processor(self, component_type):
        """Get the processor function for a component type"""
        return self.component_processors.get(component_type)
        
    def _process_basketball_tracker(self, frame):
        """Process frame with basketball tracker"""
        try:
            # Apply resize if needed
            if self.input_resize_factor != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * self.input_resize_factor), int(w * self.input_resize_factor)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                tracks, annotated_frame = self.basketball_tracker.track_frame(resized_frame)
                # Resize annotated frame back to original size
                annotated_frame = cv2.resize(annotated_frame, (w, h))
                
                # Adjust track coordinates back to original scale
                scale_factor = 1.0 / self.input_resize_factor
                for track in tracks:
                    if 'center' in track:
                        track['center'] = (track['center'][0] * scale_factor, 
                                          track['center'][1] * scale_factor)
                    if 'bbox' in track:
                        x1, y1, x2, y2 = track['bbox']
                        track['bbox'] = (x1 * scale_factor, y1 * scale_factor, 
                                        x2 * scale_factor, y2 * scale_factor)
            else:
                tracks, annotated_frame = self.basketball_tracker.track_frame(frame)
                
            # Store tracks for other components to use
            self.basketball_tracks = tracks
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return tracks, frame
            
            return tracks, annotated_frame
        except Exception as e:
            logger.error(f"Error in basketball tracker: {e}")
            return [], frame
            
    def _process_pose_tracker(self, frame):
        """Process frame with pose tracker"""
        try:
            # Apply resize if needed
            if self.input_resize_factor != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * self.input_resize_factor), int(w * self.input_resize_factor)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                tracks, annotated_frame = self.pose_tracker.infer_frame(resized_frame)
                # Resize annotated frame back to original size
                annotated_frame = cv2.resize(annotated_frame, (w, h))
                
                # Adjust track coordinates back to original scale
                scale_factor = 1.0 / self.input_resize_factor
                for track in tracks:
                    if 'keypoints' in track:
                        for i in range(len(track['keypoints'])):
                            if track['keypoints'][i] is not None:
                                x, y, conf = track['keypoints'][i]
                                track['keypoints'][i] = (x * scale_factor, y * scale_factor, conf)
                    if 'bbox' in track:
                        x1, y1, x2, y2 = track['bbox']
                        track['bbox'] = (x1 * scale_factor, y1 * scale_factor, 
                                        x2 * scale_factor, y2 * scale_factor)
            else:
                tracks, annotated_frame = self.pose_tracker.infer_frame(frame)
                
            # Store tracks for other components to use
            self.pose_tracks = tracks
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return tracks, frame
                
            return tracks, annotated_frame
        except Exception as e:
            logger.error(f"Error in pose tracker: {e}")
            return [], frame
    
    def _process_dribble_counter(self, frame):
        """Process frame with dribble counter"""
        try:
            # Use cached basketball tracks if available with None checking
            if self.basketball_tracks is not None and len(self.basketball_tracks) > 0:
                if hasattr(self.dribble_counter, 'process_frame_with_tracks'):
                    dribble_result = self.dribble_counter.process_frame_with_tracks(frame, self.basketball_tracks)
                else:
                    dribble_result = self.dribble_counter.process_frame(frame)
            else:
                dribble_result = self.dribble_counter.process_frame(frame)
            
            # Extract results
            dribble_count = self.dribble_counter.get_dribble_count() if hasattr(self.dribble_counter, 'get_dribble_count') else 0
            
            # Handle different return formats
            if isinstance(dribble_result, tuple) and dribble_result is not None:
                dribble_frame = dribble_result[0] if len(dribble_result) > 0 else frame
            elif isinstance(dribble_result, dict):
                dribble_frame = dribble_result.get('annotated_frame', frame)
            elif dribble_result is None:
                dribble_frame = frame
            else:
                dribble_frame = frame
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return dribble_count, frame
            
            return dribble_count, dribble_frame
        except Exception as e:
            logger.error(f"Error in dribble counter: {e}")
            return 0, frame
    
    def _process_holding_detector(self, frame):
        """Process frame with ball holding detector"""
        try:
            # Use cached tracks if available
            if self.basketball_tracks:
                # Check if method exists
                if hasattr(self.holding_detector, 'process_frame_with_tracks'):
                    holding_result = self.holding_detector.process_frame_with_tracks(
                        frame, self.basketball_tracks)
                else:
                    holding_result = self.holding_detector.process_frame(frame)
            else:
                holding_result = self.holding_detector.process_frame(frame)
            
            # Handle different return formats
            if isinstance(holding_result, tuple) and holding_result is not None:
                # Format: (annotated_frame, ball_detected) or (is_holding, player_id)
                if len(holding_result) == 2:
                    holding_frame, ball_detected = holding_result
                    is_holding = self.holding_detector.is_holding if hasattr(self.holding_detector, 'is_holding') else False
                    holding_player_id = -1  # Default player ID
                else:
                    is_holding, holding_player_id = False, -1
                    holding_frame = frame
            elif isinstance(holding_result, dict):
                # Dictionary format
                is_holding = holding_result.get('is_holding', False)
                holding_frame = holding_result.get('annotated_frame', frame)
                holding_player_id = holding_result.get('player_id', -1)
            elif holding_result is None:
                # Handle None result
                is_holding, holding_player_id = False, -1
                holding_frame = frame
            else:
                # Fallback
                is_holding, holding_player_id = False, -1
                holding_frame = frame
            
            # Update state
            self.is_holding_ball = is_holding
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return (is_holding, holding_player_id), frame
            
            return (is_holding, holding_player_id), holding_frame
        except Exception as e:
            logger.error(f"Error in holding detector: {e}")
            return (False, -1), frame
    
    def _process_step_counter(self, frame):
        """Process frame with step counter"""
        try:
            # Use cached pose tracks if available
            if self.pose_tracks:
                if hasattr(self.step_counter, 'process_frame_with_poses'):
                    step_result = self.step_counter.process_frame_with_poses(frame, self.pose_tracks)
                else:
                    step_result = self.step_counter.process_frame(frame)
            else:
                step_result = self.step_counter.process_frame(frame)
            
            # Handle different return formats
            if isinstance(step_result, tuple) and step_result is not None:
                step_frame = step_result[0] if len(step_result) > 0 else frame
                step_count = self.step_counter.get_step_count() if hasattr(self.step_counter, 'get_step_count') else {}
            elif isinstance(step_result, dict):
                step_count = step_result.get('step_count', {})
                step_frame = step_result.get('annotated_frame', frame)
            elif step_result is None:
                step_count = {}
                step_frame = frame
            else:
                step_count = {}
                step_frame = frame
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return step_count, frame
            
            return step_count, step_frame
        except Exception as e:
            logger.error(f"Error in step counter: {e}")
            return {}, frame
    
    def _process_shot_detector(self, frame):
        """Process frame with shot detector"""
        try:
            # Use cached basketball tracks if available with None checking
            if self.basketball_tracks is not None and len(self.basketball_tracks) > 0:
                if hasattr(self.shot_detector, 'process_frame_with_tracks'):
                    shot_result = self.shot_detector.process_frame_with_tracks(frame, self.basketball_tracks)
                else:
                    shot_result = self.shot_detector.process_frame(frame)
            else:
                shot_result = self.shot_detector.process_frame(frame)
            
            # Extract results
            shot_stats = self.shot_detector.get_statistics() if hasattr(self.shot_detector, 'get_statistics') else {'makes': 0, 'attempts': 0}
            
            # Handle different return formats
            if isinstance(shot_result, tuple) and shot_result is not None:
                shot_frame = shot_result[0] if len(shot_result) > 0 else frame
            elif isinstance(shot_result, dict):
                shot_frame = shot_result.get('annotated_frame', frame)
            elif shot_result is None:
                shot_frame = frame
            else:
                shot_frame = frame
            
            # Check for new shots
            current_attempts = shot_stats.get('attempts', 0)
            current_makes = shot_stats.get('makes', 0)
            
            # Update global statistics
            if current_attempts > self.total_shots:
                self.total_shots = current_attempts
                self.total_makes = current_makes
                logger.info(f"Updated shot statistics: {self.total_makes}/{self.total_shots}")
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return shot_stats, frame
            
            return shot_stats, shot_frame
        except Exception as e:
            logger.error(f"Error in shot detector: {e}")
            return {'makes': 0, 'attempts': 0}, frame
    
    def _process_double_dribble_detector(self, frame):
        """Process frame with double dribble detector"""
        try:
            # Use cached data if available
            if self.basketball_tracks and self.is_holding_ball:
                if hasattr(self.double_dribble_detector, 'process_frame_with_data'):
                    double_dribble_result = self.double_dribble_detector.process_frame_with_data(
                        frame, self.basketball_tracks, self.is_holding_ball)
                else:
                    double_dribble_result = self.double_dribble_detector.process_frame(frame)
            else:
                double_dribble_result = self.double_dribble_detector.process_frame(frame)
            
            # Handle different return formats
            if isinstance(double_dribble_result, tuple):
                violation_frame = double_dribble_result[0] if len(double_dribble_result) > 0 else frame
                violation_detected = double_dribble_result[1] if len(double_dribble_result) > 1 else False
            elif isinstance(double_dribble_result, dict):
                violation_detected = double_dribble_result.get('violation', False)
                violation_frame = double_dribble_result.get('annotated_frame', frame)
            else:
                violation_detected = False
                violation_frame = frame
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return violation_detected, frame
            
            return violation_detected, violation_frame
        except Exception as e:
            logger.error(f"Error in double dribble detector: {e}")
            return False, frame

    def _process_travel_detector(self, frame):
        """Process frame with travel violation detector"""
        try:
            # Use cached data if available
            if self.pose_tracks and self.is_holding_ball:
                if hasattr(self.travel_detector, 'process_frame_with_data'):
                    travel_result = self.travel_detector.process_frame_with_data(
                        frame, self.pose_tracks, self.is_holding_ball)
                else:
                    travel_result = self.travel_detector.process_frame(frame)
            else:
                travel_result = self.travel_detector.process_frame(frame)
            
            # Handle different return formats
            if isinstance(travel_result, tuple):
                travel_frame = travel_result[0] if len(travel_result) > 0 else frame
                violation_detected = travel_result[1] if len(travel_result) > 1 else False
            elif isinstance(travel_result, dict):
                violation_detected = travel_result.get('violation', False)
                travel_frame = travel_result.get('annotated_frame', frame)
            else:
                violation_detected = False
                travel_frame = frame
            
            # If visualization level is minimal, return original frame
            if self.visualization_level == 0:
                return violation_detected, frame
            
            return violation_detected, travel_frame
        except Exception as e:
            logger.error(f"Error in travel detector: {e}")
            return False, frame
    
    def _combine_annotations(self, original_frame, frames_dict):
        """Combine annotations from multiple frames efficiently"""
        # Start with the original frame
        result_frame = original_frame.copy()
        
        # If visualization level is minimal, return original frame
        if self.visualization_level == 0:
            return result_frame
        
        # Combine annotations based on visualization level
        if self.visualization_level >= 1:
            # Add critical components (basketball and pose tracking)
            if ComponentType.BASKETBALL_TRACKER in frames_dict:
                basketball_frame = frames_dict[ComponentType.BASKETBALL_TRACKER]
                if basketball_frame is not None and isinstance(basketball_frame, np.ndarray):
                    result_frame = cv2.addWeighted(result_frame, 0.7, basketball_frame, 0.3, 0)
            if ComponentType.POSE_TRACKER in frames_dict:
                pose_frame = frames_dict[ComponentType.POSE_TRACKER]
                if pose_frame is not None and isinstance(pose_frame, np.ndarray):
                    result_frame = cv2.addWeighted(result_frame, 0.7, pose_frame, 0.3, 0)
        
        if self.visualization_level >= 2:
            # Add all other components
            for component_type, frame in frames_dict.items():
                if component_type not in [ComponentType.BASKETBALL_TRACKER, ComponentType.POSE_TRACKER]:
                    if frame is not None and isinstance(frame, np.ndarray):
                        result_frame = cv2.addWeighted(result_frame, 0.8, frame, 0.2, 0)
        
        return result_frame
    
    def reset(self):
        """Reset all detectors and statistics"""
        self.dribble_counter.reset_counter()
        self.travel_detector.reset()
        self.double_dribble_detector.reset()
        self.shot_detector.reset_counter()
        self.statistics = GameStatistics()
        self.frame_count = 0
        self.last_violation_time = 0
        self.component_results = {}
        self.basketball_tracks = []
        self.pose_tracks = []
        self.current_violations = []
        logger.info("Reset all detectors and statistics")
    
    def process_frame(self, frame):
        """
        Process a single frame with all basketball referee components using asynchronous processing
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with all referee components
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Calculate current time based on frame count and FPS
        current_time = self.frame_count / max(1, self.fps)
        
        # Create a copy of the frame for processing
        original_frame = frame.copy()
        
        # Dictionary to store annotated frames from components
        annotated_frames = {}
        
        # Dictionary to store component results
        results = {}
        
        # Track if any violations occurred in this frame
        new_violations = []
        
        # Process critical components first (basketball and pose tracking)
        # These are always processed synchronously for every frame
        if self.frame_count % self.frame_interval[ComponentType.BASKETBALL_TRACKER] == 0:
            basketball_result, basketball_frame = self._process_basketball_tracker(frame)
            results[ComponentType.BASKETBALL_TRACKER] = basketball_result
            annotated_frames[ComponentType.BASKETBALL_TRACKER] = basketball_frame
        
        if self.frame_count % self.frame_interval[ComponentType.POSE_TRACKER] == 0:
            pose_result, pose_frame = self._process_pose_tracker(frame)
            results[ComponentType.POSE_TRACKER] = pose_result
            annotated_frames[ComponentType.POSE_TRACKER] = pose_frame
        
        # Process ball holding detection (critical component)
        if self.frame_count % self.frame_interval[ComponentType.HOLDING_DETECTOR] == 0:
            holding_result, holding_frame = self._process_holding_detector(frame)
            is_holding, holding_player_id = holding_result
            results[ComponentType.HOLDING_DETECTOR] = holding_result
            annotated_frames[ComponentType.HOLDING_DETECTOR] = holding_frame
        else:
            # Use cached result
            holding_result = self.component_results.get(ComponentType.HOLDING_DETECTOR, (False, -1))
            is_holding, holding_player_id = holding_result
        
        # Asynchronous processing for non-critical components
        async_tasks = {}
        
        # Submit asynchronous tasks based on frame intervals
        if self.async_processing:
            # Dribble counter
            if self.frame_count % self.frame_interval[ComponentType.DRIBBLE_COUNTER] == 0:
                task = ProcessingTask(ComponentType.DRIBBLE_COUNTER, frame.copy(), self.frame_count, 3)
                self.task_processor.submit_task(task)
            
            # Step counter
            if self.frame_count % self.frame_interval[ComponentType.STEP_COUNTER] == 0:
                task = ProcessingTask(ComponentType.STEP_COUNTER, frame.copy(), self.frame_count, 2)
                self.task_processor.submit_task(task)
            
            # Shot detector
            if self.frame_count % self.frame_interval[ComponentType.SHOT_DETECTOR] == 0:
                task = ProcessingTask(ComponentType.SHOT_DETECTOR, frame.copy(), self.frame_count, 2)
                self.task_processor.submit_task(task)
            
            # Travel detector
            if is_holding and self.frame_count % self.frame_interval[ComponentType.TRAVEL_DETECTOR] == 0:
                task = ProcessingTask(ComponentType.TRAVEL_DETECTOR, frame.copy(), self.frame_count, 1)
                self.task_processor.submit_task(task)
            
            # Double dribble detector
            if self.frame_count % self.frame_interval[ComponentType.DOUBLE_DRIBBLE_DETECTOR] == 0:
                task = ProcessingTask(ComponentType.DOUBLE_DRIBBLE_DETECTOR, frame.copy(), self.frame_count, 1)
                self.task_processor.submit_task(task)
            
            # Collect results from async tasks with timeout
            timeout_sec = 0.05  # 50ms timeout
            async_results = self.task_processor.get_results(timeout_sec)
            
            # Iterate over the dictionary items
            for component_type, result in async_results.items():
                component_result = result.result
                component_frame = result.annotated_frame
                
                # Store results and annotated frames
                results[component_type] = component_result
                annotated_frames[component_type] = component_frame
                
                # Cache results for future frames
                self.component_results[component_type] = component_result
        else:
            # Synchronous processing for all components
            # Dribble counter
            if self.frame_count % self.frame_interval[ComponentType.DRIBBLE_COUNTER] == 0:
                dribble_result, dribble_frame = self._process_dribble_counter(frame)
                results[ComponentType.DRIBBLE_COUNTER] = dribble_result
                annotated_frames[ComponentType.DRIBBLE_COUNTER] = dribble_frame
            
            # Step counter
            if self.frame_count % self.frame_interval[ComponentType.STEP_COUNTER] == 0:
                step_result, step_frame = self._process_step_counter(frame)
                results[ComponentType.STEP_COUNTER] = step_result
                annotated_frames[ComponentType.STEP_COUNTER] = step_frame
            
            # Shot detector
            if self.frame_count % self.frame_interval[ComponentType.SHOT_DETECTOR] == 0:
                shot_result, shot_frame = self._process_shot_detector(frame)
                results[ComponentType.SHOT_DETECTOR] = shot_result
                annotated_frames[ComponentType.SHOT_DETECTOR] = shot_frame
            
            # Travel detector
            if is_holding and self.frame_count % self.frame_interval[ComponentType.TRAVEL_DETECTOR] == 0:
                travel_result, travel_frame = self._process_travel_detector(frame)
                results[ComponentType.TRAVEL_DETECTOR] = travel_result
                annotated_frames[ComponentType.TRAVEL_DETECTOR] = travel_frame
            
            # Double dribble detector
            if self.frame_count % self.frame_interval[ComponentType.DOUBLE_DRIBBLE_DETECTOR] == 0:
                double_dribble_result, double_dribble_frame = self._process_double_dribble_detector(frame)
                results[ComponentType.DOUBLE_DRIBBLE_DETECTOR] = double_dribble_result
                annotated_frames[ComponentType.DOUBLE_DRIBBLE_DETECTOR] = double_dribble_frame
        
        # Use cached results for components that weren't processed this frame
        for component_type in ComponentType:
            if component_type not in results and component_type in self.component_results:
                results[component_type] = self.component_results[component_type]
        
        # Check for violations
        # Travel violation
        if ComponentType.TRAVEL_DETECTOR in results and results[ComponentType.TRAVEL_DETECTOR]:
            if current_time - self.last_violation_time > self.violation_cooldown:
                new_violations.append("Travel Violation")
                self.statistics.add_violation("travel")
                self.last_violation_time = current_time
        
        # Double dribble violation
        if ComponentType.DOUBLE_DRIBBLE_DETECTOR in results and results[ComponentType.DOUBLE_DRIBBLE_DETECTOR]:
            if current_time - self.last_violation_time > self.violation_cooldown:
                new_violations.append("Double Dribble Violation")
                self.statistics.add_violation("double_dribble")
                self.last_violation_time = current_time
        
        # Update statistics with debug logging
        if ComponentType.DRIBBLE_COUNTER in results:
            dribble_result = results[ComponentType.DRIBBLE_COUNTER]
            if isinstance(dribble_result, dict):
                dribble_count = dribble_result.get('dribble_count', 0)
            else:
                dribble_count = self.dribble_counter.get_dribble_count() if hasattr(self.dribble_counter, 'get_dribble_count') else 0
            self.statistics.update_dribble_count(dribble_count)
            logger.info(f"Updated dribble count to: {dribble_count}")
        else:
            # Try to get dribble count directly from detector
            if hasattr(self.dribble_counter, 'get_dribble_count'):
                dribble_count = self.dribble_counter.get_dribble_count()
                if dribble_count > 0:
                    self.statistics.update_dribble_count(dribble_count)
                    logger.info(f"Direct update dribble count to: {dribble_count}")
        
        if ComponentType.STEP_COUNTER in results:
            step_result = results[ComponentType.STEP_COUNTER]
            logger.info(f"Step counter result: {step_result}")
            if isinstance(step_result, dict):
                # Check if it has 'step_counts' key (the actual format from step counter)
                if 'step_counts' in step_result:
                    step_counts = step_result['step_counts']
                    for player_id, steps in step_counts.items():
                        if isinstance(player_id, int) and isinstance(steps, int):
                            self.statistics.step_count[player_id] = steps
                            logger.info(f"Updated step count for player {player_id}: {steps}")
                else:
                    # Direct dictionary format
                    for player_id, steps in step_result.items():
                        if isinstance(player_id, int) and isinstance(steps, int):
                            self.statistics.step_count[player_id] = steps
                            logger.info(f"Updated step count for player {player_id}: {steps}")
            else:
                # Get step count from step counter directly
                step_counts = self.step_counter.get_step_count() if hasattr(self.step_counter, 'get_step_count') else {}
                if isinstance(step_counts, dict):
                    for player_id, steps in step_counts.items():
                        self.statistics.step_count[player_id] = steps
                        logger.info(f"Direct update step count for player {player_id}: {steps}")
        else:
            logger.info("ComponentType.STEP_COUNTER not in results, trying direct access")
            # Try to get step count directly from detector
            if hasattr(self.step_counter, 'person_data'):
                person_data = getattr(self.step_counter, 'person_data', {})
                logger.info(f"Step counter person_data: {person_data}")
                if isinstance(person_data, dict) and person_data:
                    for person_id, data in person_data.items():
                        if isinstance(data, dict) and 'step_count' in data:
                            steps = data['step_count']
                            if steps > 0:
                                self.statistics.step_count[person_id] = steps
                                logger.info(f"Direct access update step count for player {person_id}: {steps}")
            elif hasattr(self.step_counter, 'get_step_count'):
                step_counts = self.step_counter.get_step_count()
                logger.debug(f"Step counter get_step_count returned: {step_counts}")
                if isinstance(step_counts, dict) and step_counts:
                    for player_id, steps in step_counts.items():
                        if steps > 0:
                            self.statistics.step_count[player_id] = steps
                            logger.info(f"Fallback update step count for player {player_id}: {steps}")
                else:
                    # Try alternative method to get step counts
                    if hasattr(self.step_counter, 'step_counts'):
                        alt_step_counts = getattr(self.step_counter, 'step_counts', {})
                        logger.debug(f"Alternative step counts: {alt_step_counts}")
                        if isinstance(alt_step_counts, dict) and alt_step_counts:
                            for player_id, steps in alt_step_counts.items():
                                if steps > 0:
                                    self.statistics.step_count[player_id] = steps
                                    logger.info(f"Alternative update step count for player {player_id}: {steps}")
        
        if ComponentType.SHOT_DETECTOR in results:
            shot_stats = results[ComponentType.SHOT_DETECTOR]
            # Handle different formats of shot statistics
            try:
                if isinstance(shot_stats, dict):
                    # Direct dictionary format
                    makes = shot_stats.get('makes', 0)
                    attempts = shot_stats.get('attempts', 0)
                    self.statistics.set_shot_stats(makes, attempts)
                elif hasattr(shot_stats, 'data'):
                    # ProcessingResult with data attribute
                    data = shot_stats.data
                    if isinstance(data, dict):
                        makes = data.get('makes', 0)
                        attempts = data.get('attempts', 0)
                    else:
                        makes = getattr(data, 'makes', 0)
                        attempts = getattr(data, 'attempts', 0)
                    self.statistics.set_shot_stats(makes, attempts)
                elif hasattr(shot_stats, 'makes') and hasattr(shot_stats, 'attempts'):
                    # Object with direct attributes
                    makes = shot_stats.makes
                    attempts = shot_stats.attempts
                    self.statistics.set_shot_stats(makes, attempts)
                else:
                    # Fallback to extracting from shot detector
                    detector_stats = self.shot_detector.get_statistics()
                    if isinstance(detector_stats, dict):
                        makes = detector_stats.get('makes', 0)
                        attempts = detector_stats.get('attempts', 0)
                    else:
                        makes = getattr(detector_stats, 'makes', 0)
                        attempts = getattr(detector_stats, 'attempts', 0)
                    self.statistics.set_shot_stats(makes, attempts)
                    
                logger.debug(f"Updated shot stats: makes={makes}, attempts={attempts}")
            except Exception as e:
                logger.error(f"Error processing shot statistics: {e}")
                self.statistics.set_shot_stats(0, 0)
        
        # Combine all annotations onto one frame
        result_frame = self._combine_annotations(original_frame, annotated_frames)
        
        # Add violations overlay
        if self.show_violations and new_violations:
            self.current_violations.extend([(v, current_time) for v in new_violations])
        
        # Remove expired violations
        self.current_violations = [(v, t) for v, t in self.current_violations 
                                 if current_time - t <= self.violation_display_time]
        
        # Display current violations
        if self.show_violations and self.current_violations:
            y_offset = 50
            for violation, _ in self.current_violations:
                cv2.putText(result_frame, violation, (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                y_offset += 40
        
        # Add statistics overlay
        if self.show_stats:
            stats_text = self.statistics.get_stats_text()
            y_offset = 50
            for line in stats_text.split('\n'):
                cv2.putText(result_frame, line, (result_frame.shape[1] - 300, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
        
        # Calculate processing time and FPS
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only the last 30 processing times for FPS calculation
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # Calculate FPS
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / max(avg_processing_time, 0.001)
        self.fps_history.append(fps)
        
        # Keep only the last 10 FPS values for smoothing
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)
        
        # Display FPS - DISABLED
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        # cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame
    
    def run(self, max_frames=None, display=True, output_path=None):
        """
        Run the basketball referee system on the video source
        
        Args:
            max_frames: Maximum number of frames to process (None for all)
            display: Whether to display the output frames
            output_path: Path to save the output video (None for no saving)
            
        Returns:
            Statistics from the basketball game
        """
        logger.info("Running Basketball Referee System")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps,
                                   (self.frame_width, self.frame_height))
        
        # Reset counters and statistics
        self.reset()
        
        # Start processing frames
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Check if we've reached the maximum number of frames
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # Read frame from video source
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Process frame
                result_frame = self.process_frame(frame)
                
                # Write frame to output video
                if writer:
                    writer.write(result_frame)
                
                # Display frame
                if display:
                    cv2.imshow("Basketball Referee", result_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User pressed 'q', exiting")
                        break
                    elif key == ord('r'):
                        logger.info("User pressed 'r', resetting statistics")
                        self.reset()
                    elif key == ord('v'):
                        # Toggle visualization level
                        self.visualization_level = (self.visualization_level + 1) % 3
                        logger.info(f"Visualization level set to {self.visualization_level}")
                    elif key == ord('a'):
                        # Toggle async processing
                        self.async_processing = not self.async_processing
                        logger.info(f"Async processing {'enabled' if self.async_processing else 'disabled'}")
                
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Processed {frame_count} frames at {fps:.2f} FPS")
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, stopping")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # Release resources
            if writer:
                writer.release()
            
            if display:
                cv2.destroyAllWindows()
            
            # Stop task processor
            if self.async_processing:
                self.task_processor.stop()
            
            logger.info(f"Processed {frame_count} frames")
            
        return self.statistics
    
    def get_statistics(self):
        """
        Get current game statistics
        
        Returns:
            Dictionary with game statistics
        """
        return self.statistics.get_stats()
    
    def get_violations(self):
        """
        Get current violations
        
        Returns:
            List of current violations
        """
        return [v for v, _ in self.current_violations]
    
    def stop(self):
        """
        Stop the basketball referee system and release resources
        """
        logger.info("Stopping Basketball Referee System")
        
        # Release video capture
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
        # Stop task processor
        if hasattr(self, 'task_processor') and self.async_processing:
            self.task_processor.stop()
        
        # Clean up OpenCV windows
        cv2.destroyAllWindows()



def main():
    """Main entry point"""
    try:
        import argparse
       
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='AI Basketball Referee System')
        parser.add_argument('--video', type=str, default='data/video/parallel_angle.mov',
                          help='Path to input video file')
        parser.add_argument('--basketball-model', type=str, 
                          default="models/ov_models/basketballModel_openvino_model/basketballModel.xml",
                          help='Path to basketball detection model')
        parser.add_argument('--pose-model', type=str,
                          default="models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml",
                          help='Path to pose detection model')
        args = parser.parse_args()
        
        # Configuration
        basketball_model = args.basketball_model
        pose_model = args.pose_model
        video_path = args.video
        
        # Check if models exist
        if not os.path.exists(basketball_model):
            logger.error(f"Basketball model not found: {basketball_model}")
            return
        
        if not os.path.exists(pose_model):
            logger.error(f"Pose model not found: {pose_model}")
            return
        
        # Initialize and run referee system
        referee = BasketballReferee(
            basketball_model_path=basketball_model,
            pose_model_path=pose_model,
            video_path=video_path,
            device=DeviceType.CPU,
            rules="FIBA"
        )
        
        referee.run()
        
        # Print final statistics
        stats = referee.get_statistics()
        
        # Get shot statistics directly from shot detector as backup
        shot_stats = referee.shot_detector.get_statistics()
        shot_attempts = shot_stats.get('attempts', 0) if isinstance(shot_stats, dict) else getattr(shot_stats, 'attempts', 0)
        shot_makes = shot_stats.get('makes', 0) if isinstance(shot_stats, dict) else getattr(shot_stats, 'makes', 0)
        
        # Use shot detector stats if main stats are zero
        if stats['shot_attempts'] == 0 and shot_attempts > 0:
            stats['shot_attempts'] = shot_attempts
            stats['shot_makes'] = shot_makes
            stats['shooting_percentage'] = (shot_makes / shot_attempts * 100) if shot_attempts > 0 else 0
        
        print("\n===== FINAL GAME STATISTICS =====")
        print(f"Shots: {stats['shot_makes']}/{stats['shot_attempts']} ({stats['shooting_percentage']:.1f}%)")
        print(f"Dribbles: {stats['dribble_count']}")
        
        # Calculate total steps from step_count dictionary
        total_steps = sum(stats['step_count'].values()) if stats['step_count'] else 0
        print(f"Total Steps: {total_steps}")
        
        # Show individual player steps if available
        if stats['step_count']:
            for player_id, steps in stats['step_count'].items():
                print(f"  Player {player_id}: {steps} steps")
        
        print(f"Violations: {len(stats['violations'])}")
        for i, v in enumerate(stats['violations']):
            print(f"  {i+1}. {v['type'].upper()}: {v.get('description', 'No description')}")
        
    except Exception as e:
        logger.error(f"Error running Basketball Referee System: {e}")


if __name__ == "__main__":
    main()