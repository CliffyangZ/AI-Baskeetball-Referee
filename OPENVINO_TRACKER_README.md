# OpenVINO Basketball and Pose Trackers

This project implements high-performance basketball and pose tracking using OpenVINO runtime inference with the BYTETrack algorithm for multi-object tracking.

## Features

### Basketball Tracker (`basketballTracker.py`)
- **OpenVINO optimized inference** for fast basketball detection
- **BYTETrack algorithm** for robust multi-object tracking
- **Two-stage association** with high/low confidence thresholds
- **Kalman filtering** for smooth trajectory prediction
- **Configurable parameters** for different scenarios
- **Multi-device support** (CPU, GPU, NPU, AUTO)

### Pose Tracker (`poseTracker.py`)
- **Multiple pose model support** (YOLOv8-Pose, OpenPose, MoveNet, HRNet)
- **17-keypoint COCO format** pose estimation
- **Multi-person tracking** with pose similarity matching
- **Skeleton visualization** with customizable colors
- **Keypoint confidence filtering** for robust detection
- **Bounding box generation** from visible keypoints

## Installation

### Prerequisites
```bash
# Install OpenVINO (if not already installed)
pip install openvino==2025.2.0

# Install other dependencies
pip install -r requirements.txt
```

### Required Dependencies
- `openvino>=2025.2.0` - OpenVINO runtime
- `opencv-python>=4.12.0` - Computer vision operations
- `numpy>=2.2.0` - Numerical computations
- `filterpy>=1.4.5` - Kalman filtering (optional)

## Usage

### Basic Usage

#### Basketball Tracking
```python
from tracker.basketballTracker import BasketballTracker, DeviceType

# Initialize tracker
tracker = BasketballTracker(
    model_path="path/to/basketball_model.xml",
    device=DeviceType.CPU,
    high_thresh=0.6,
    low_thresh=0.1,
    match_thresh=0.8
)

# Process frame
tracks, annotated_frame = tracker.infer_frame(frame)
```

#### Pose Tracking
```python
from tracker.poseTracker import PoseTracker, DeviceType, PoseModel

# Initialize tracker
tracker = PoseTracker(
    model_path="path/to/pose_model.xml",
    device=DeviceType.CPU,
    model_type=PoseModel.YOLOV8_POSE,
    confidence_threshold=0.3
)

# Process frame
tracks, annotated_frame = tracker.infer_frame(frame)
```

### Command Line Demo

The project includes a comprehensive demo script:

```bash
# Basketball tracking with webcam
python openvino_tracker_example.py --mode basketball --model models/basketball.xml --source 0

# Pose tracking with video file
python openvino_tracker_example.py --mode pose --model models/pose.xml --source video.mp4

# Process single image
python openvino_tracker_example.py --mode pose --model models/pose.xml --source image.jpg --image

# Show device information
python openvino_tracker_example.py --mode basketball --model models/basketball.xml --info
```

### Advanced Configuration

#### Basketball Tracker Parameters
```python
tracker = BasketballTracker(
    model_path="basketball_model.xml",
    device=DeviceType.CPU,
    high_thresh=0.6,        # High confidence threshold for first association
    low_thresh=0.1,         # Low confidence threshold for second association
    match_thresh=0.8,       # IoU threshold for track association
    max_time_lost=30        # Maximum frames to keep lost tracks
)
```

#### Pose Tracker Parameters
```python
tracker = PoseTracker(
    model_path="pose_model.xml",
    device=DeviceType.CPU,
    model_type=PoseModel.YOLOV8_POSE,
    confidence_threshold=0.3,   # Minimum confidence for pose detection
    max_time_lost=30           # Maximum frames to keep lost tracks
)
```

## OpenVINO Model Conversion

### Converting PyTorch Models to OpenVINO

#### Basketball Detection Model (YOLO)
```python
import openvino as ov
from ultralytics import YOLO

# Load PyTorch model
model = YOLO("basketball_yolo.pt")

# Export to OpenVINO IR format
model.export(format="openvino", dynamic=False, half=False)
```

#### Pose Estimation Model
```python
# For YOLOv8-Pose
model = YOLO("yolov8n-pose.pt")
model.export(format="openvino")

# For custom PyTorch pose models
import torch
import openvino as ov

# Convert PyTorch model
ov_model = ov.convert_model(pytorch_model, example_input=dummy_input)
ov.save_model(ov_model, "pose_model.xml")
```

## BYTETrack Algorithm Implementation

The basketball tracker implements the BYTETrack algorithm with the following stages:

### Stage 1: High-Confidence Association
- Match high-confidence detections (>= `high_thresh`) with active tracks
- Use IoU-based association with Hungarian algorithm
- Update matched tracks and mark unmatched tracks as candidates for Stage 2

### Stage 2: Low-Confidence Recovery
- Associate remaining high-confidence detections with lost tracks
- Recover tracks that were temporarily lost due to occlusion

### Stage 3: Low-Confidence Association
- Match low-confidence detections (`low_thresh` to `high_thresh`) with unmatched tracks
- Helps maintain tracks during partial occlusions

### Key Benefits
- **Robust tracking** during occlusions and appearance changes
- **Reduced identity switches** compared to traditional methods
- **Framework agnostic** - works with any detector
- **Configurable thresholds** for different scenarios

## Performance Optimization

### Device Selection
```python
# CPU - Good compatibility, moderate performance
device = DeviceType.CPU

# GPU - Better performance for large models (requires Intel GPU)
device = DeviceType.GPU

# NPU - Best efficiency for supported models (Intel NPU required)
device = DeviceType.NPU

# AUTO - Automatically select best available device
device = DeviceType.AUTO
```

### Model Optimization Tips
1. **Use FP16 precision** for better performance on GPU/NPU
2. **Optimize input resolution** based on accuracy requirements
3. **Batch processing** for multiple frames (if applicable)
4. **Model quantization** for edge deployment

### Performance Monitoring
```python
# Get device information
device_info = tracker.get_device_info()
print(f"Available devices: {device_info['available_devices']}")
print(f"Current device: {device_info['current_device']}")
```

## Integration with Existing UI

The trackers are compatible with the existing Flet UI through wrapper classes:

```python
# For basketball tracking
from tracker.basketballTracker import OptimizedBasketballModel

model = OptimizedBasketballModel("basketball_model.xml", device="CPU")
annotated_frame = model.infer_frame(frame)

# For pose tracking
from tracker.poseTracker import OptimizedPoseModel

model = OptimizedPoseModel("pose_model.xml", device="CPU", model_type="yolov8_pose")
annotated_frame = model.infer_frame(frame)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure `.xml` and `.bin` files are in the same directory
   - Check OpenVINO version compatibility
   - Verify model format (should be OpenVINO IR)

2. **Performance Issues**
   - Try different devices (CPU/GPU/NPU)
   - Reduce input resolution
   - Adjust confidence thresholds

3. **Tracking Issues**
   - Tune `match_thresh` for your scenario
   - Adjust `high_thresh` and `low_thresh` based on model performance
   - Increase `max_time_lost` for longer occlusions

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
```

## Model Requirements

### Basketball Detection Model
- **Input**: RGB image (NCHW format)
- **Output**: Detections with format `[x1, y1, x2, y2, confidence, class_id]`
- **Classes**: Basketball (class_id = 0)

### Pose Estimation Model
- **Input**: RGB image (NCHW format)
- **Output**: 
  - YOLOv8-Pose: `[batch, 56, 8400]` where 56 = 4(bbox) + 1(conf) + 51(17 keypoints × 3)
  - OpenPose: `[batch, 57, H, W]` where 57 = 18 keypoints + 19 PAFs
- **Keypoints**: 17 COCO format keypoints

## Contributing

When contributing to the tracker implementations:

1. **Maintain OpenVINO compatibility** across versions
2. **Follow BYTETrack algorithm** specifications
3. **Add comprehensive logging** for debugging
4. **Include performance benchmarks** for new features
5. **Update documentation** for API changes

## License

This implementation follows the project's existing license terms.
