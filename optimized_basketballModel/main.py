from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import yaml
import sys
import os

from EnhancedModel import EnhancedBasketballTracker
from BasketballTracker import BasketballTracker
from PerformanceMonitor import PerformanceMonitor

def main():
    """
    Main function to demonstrate basketball tracking.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Basketball Tracking Demo')
    parser.add_argument('--model', type=str, default='../pt_models/basketballModel.pt', help='Path to the YOLO model')
    parser.add_argument('--source', type=str, default='../data/video/dribbling.mov', help='Path to the video file')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced tracking with Kalman filtering')
    parser.add_argument('--scale', type=float, default=1.0, help='Display scale factor')
    args = parser.parse_args()
    
    print(f"Basketball Tracking Demo")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Enhanced Mode: {'Yes' if args.enhanced else 'No'}")
   
    try:
        # Create tracker based on mode
        if args.enhanced:
            tracker = EnhancedBasketballTracker(args.model)
            print("Using Enhanced Basketball Tracker with Kalman filtering")
        else:
            tracker = BasketballTracker(args.model)
            print("Using Standard Basketball Tracker")
        
        # Determine if source is a webcam or video file
        if args.source.isdigit():
            print(f"Starting real-time tracking with webcam {args.source}")
            tracker.track_realtime(int(args.source), args.scale)
        else:
            print(f"Processing video file: {args.source}")
            tracker.track_video(args.source)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
