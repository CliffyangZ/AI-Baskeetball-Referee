import time
from collections import deque

class PerformanceMonitor:
    """
    Monitor and report on tracking performance metrics.
    """
    def __init__(self, window_size=100):
        """
        Initialize the performance monitor.
        
        Args:
            window_size (int): Number of frames to keep in history
        """
        self.fps_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.frame_start_time = 0
    
    def reset(self):
        """
        Reset all performance metrics.
        """
        self.fps_history.clear()
        self.processing_times.clear()
    
    def start_frame(self):
        """
        Start timing a new frame.
        """
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """
        End timing for the current frame and calculate FPS.
        
        Returns:
            float: Current frames per second
        """
        processing_time = time.time() - self.frame_start_time
        self.processing_times.append(processing_time)
        
        fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_history.append(fps)
        
        return fps
    
    def get_stats(self):
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        if not self.fps_history:
            return {'avg_fps': 0, 'avg_processing_time': 0, 'min_fps': 0, 'max_fps': 0}
        
        return {
            'avg_fps': sum(self.fps_history) / len(self.fps_history),
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times),
            'min_fps': min(self.fps_history),
            'max_fps': max(self.fps_history)
        }