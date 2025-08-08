# Basketball Shot Counter with Automatic Hoop Detection

This program automatically detects basketball shots and counts successful baskets using computer vision and deep learning.

## Features

- Automatic basketball hoop detection using YOLOv8
- Basketball detection using the optimized basketball model
- Shot trajectory visualization
- Real-time shot counting
- Performance monitoring with FPS display
- Support for both video files and live camera feeds

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Ultralytics YOLO (version 8.3.176 or later)
- Pre-trained basketball model (`basketballModel.pt`)

## Usage

### Process a video file

```bash
python shot_counter_auto.py --video /path/to/basketball_video.mp4
```

### Use real-time camera feed

```bash
python shot_counter_auto.py
```

### Specify custom models

```bash
python shot_counter_auto.py --video /path/to/video.mp4 --ball-model /path/to/basketballModel.pt --hoop-model /path/to/hoopModel.pt
```

## How It Works

1. **Hoop Detection**: The program uses YOLOv8 to automatically detect the basketball hoop in the video frames. It looks for objects classified as "hoop", "rim", "basket", or "backboard".

2. **Ball Tracking**: The optimized basketball model detects the basketball in each frame.

3. **Shot Detection**: A shot is counted when:
   - The ball is detected above the hoop
   - The ball passes through the hoop area
   - The ball continues below the hoop

4. **Visualization**: The program displays:
   - The detected hoop (yellow circle)
   - The detected basketball (red circle)
   - Ball trajectory (blue line)
   - Shot count
   - FPS information

## Controls

- Press 'q' to quit the program

## Output

When processing a video file, the program saves an output video with all visualizations to a file named `shot_counter_output_[original_filename].mp4`.

## Troubleshooting

- If the hoop is not being detected, try adjusting the lighting or angle of the camera.
- For better performance, ensure the basketball and hoop are clearly visible in the frame.
- The first time you run the program, it may take some time to download the YOLOv8 model if not already present.
