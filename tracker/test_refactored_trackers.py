#!/usr/bin/env python3
"""
Test script for refactored basketball and pose trackers
Validates that the OpenVINO utils integration works correctly
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add tracker directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trackers
from basketballTracker import BasketballTracker, OptimizedBasketballModel
from poseTracker import PoseTracker, OptimizedPoseModel
from utils.openvino_utils import DeviceType

def test_basketball_tracker():
    """Test basketball tracker initialization and basic functionality"""
    print("Testing Basketball Tracker...")
    
    # Test model paths (adjust as needed)
    model_path = "models/ov_models/basketballModel_openvino_model/basketballModel.xml"
    
    if not Path(model_path).exists():
        print(f"⚠️  Basketball model not found at: {model_path}")
        print("   Skipping basketball tracker test")
        return False
    
    try:
        # Test direct tracker
        tracker = BasketballTracker(model_path, DeviceType.CPU)
        print("✅ BasketballTracker initialized successfully")
        
        # Test optimized wrapper
        opt_model = OptimizedBasketballModel(model_path, "CPU")
        print("✅ OptimizedBasketballModel initialized successfully")
        
        # Test with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test inference
        tracks, annotated_frame = tracker.infer_frame(dummy_frame)
        print(f"✅ Basketball inference successful - {len(tracks)} tracks detected")
        
        # Test coordinate extraction
        coords = tracker.get_basketball_coordinates(dummy_frame)
        print(f"✅ Basketball coordinates extracted - {len(coords)} basketballs")
        
        return True
        
    except Exception as e:
        print(f"❌ Basketball tracker test failed: {e}")
        return False

def test_pose_tracker():
    """Test pose tracker initialization and basic functionality"""
    print("\nTesting Pose Tracker...")
    
    # Test model paths (adjust as needed)
    model_path = "models/ov_models/yolov8s-pose_openvino_model/yolov8s-pose.xml"
    
    if not Path(model_path).exists():
        print(f"⚠️  Pose model not found at: {model_path}")
        print("   Skipping pose tracker test")
        return False
    
    try:
        # Test direct tracker
        from poseTracker import PoseModel
        tracker = PoseTracker(model_path, DeviceType.CPU, PoseModel.YOLOV8_POSE)
        print("✅ PoseTracker initialized successfully")
        
        # Test optimized wrapper
        opt_model = OptimizedPoseModel(model_path, "CPU")
        print("✅ OptimizedPoseModel initialized successfully")
        
        # Test with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test inference
        poses, annotated_frame = tracker.infer_frame(dummy_frame)
        print(f"✅ Pose inference successful - {len(poses)} poses detected")
        
        # Test pose extraction
        pose_info = tracker.get_human_poses(dummy_frame)
        print(f"✅ Pose information extracted - {len(pose_info)} persons")
        
        return True
        
    except Exception as e:
        print(f"❌ Pose tracker test failed: {e}")
        return False

def test_utils_integration():
    """Test OpenVINO utils integration"""
    print("\nTesting OpenVINO Utils Integration...")
    
    try:
        from utils.openvino_utils import (
            DeviceType, OpenVINOInferenceEngine, FPSCounter,
            normalize_coordinates, ensure_frame_bounds, 
            draw_fps_info, draw_detection_count, draw_detection_info
        )
        
        print("✅ All utils imported successfully")
        
        # Test FPS counter
        fps_counter = FPSCounter()
        fps = fps_counter.update()
        print(f"✅ FPS counter working - FPS: {fps}")
        
        # Test coordinate normalization
        coords = normalize_coordinates((0.5, 0.5, 0.2, 0.3), (640, 480), (1280, 960))
        print(f"✅ Coordinate normalization working - {coords}")
        
        # Test frame bounds
        bounded = ensure_frame_bounds((1300, 500), (1280, 960))
        print(f"✅ Frame bounds working - {bounded}")
        
        # Test drawing functions
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame = draw_fps_info(test_frame, 30.0)
        test_frame = draw_detection_count(test_frame, 5, "Test")
        print("✅ Drawing functions working")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Refactored Basketball and Pose Trackers")
    print("=" * 60)
    
    results = []
    
    # Test utils integration first
    results.append(test_utils_integration())
    
    # Test basketball tracker
    results.append(test_basketball_tracker())
    
    # Test pose tracker
    results.append(test_pose_tracker())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("🎉 Refactored trackers are working correctly!")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("🔧 Please check the failed components")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
