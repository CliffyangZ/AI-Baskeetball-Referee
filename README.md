# AI Basketball Referee

An AI-assisted toolkit to analyze basketball video, detect the ball and players, and assist with rule-related events such as steps, dribbles, and traveling. The system uses Ultralytics YOLO for detection, OpenCV for video processing, and Kalman filtering for tracking and smoothing.

## Requirements
- Python 3.9+ recommended
- Key libraries used in this repo:
  - ultralytics (YOLO)
  - opencv-python
  - numpy
  - scipy (e.g., `scipy.signal` in step counting)
  - pyyaml
  - gTTS, playsound (for audio feedback in some modules)

## Install the basics:
```bash
pip install -r requirements.txt
```

### Model Link:
https://drive.google.com/drive/folders/1pwZrCnAAC32OwL0ffWlkGQDtJnaZ_GX3?usp=drive_link
