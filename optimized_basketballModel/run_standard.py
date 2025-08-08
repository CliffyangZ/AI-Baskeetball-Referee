#!/usr/bin/env python3
"""
Simple script to run the standard basketball model without enhancements.
"""
import os
import sys
from BasketballTracker import BasketballTracker

def main():
    # Get the absolute path to the model and video
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_dir, "pt_models/basketballModel.pt")
    video_path = os.path.join(project_dir, "data/video/dribbling.mov")
    
    print(f"Running standard basketball tracker")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    
    # Create the standard tracker
    tracker = BasketballTracker(model_path)
    
    # Process the video
    tracker.track_video(video_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
