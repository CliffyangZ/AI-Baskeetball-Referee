AI Basketball Referee - Docker Quickstart
=========================================

This container runs the Flet web UI (frontend) and backend inference in one process.

Prerequisites
-------------
- Docker Desktop (macOS/Windows) or Docker Engine (Linux)
- For real webcam access:
  - Linux: container can access /dev/video0
  - macOS/Windows: Docker Desktop cannot pass host webcams into Linux containers; use a video file or network stream instead

Build
-----
```bash
# From repository root
docker compose build
```

Run
---
```bash
# Default: tries VIDEO_SOURCE=0 (Linux webcam). On macOS/Windows, set a file or RTSP URL instead.
VIDEO_SOURCE=0 DEVICE=CPU docker compose up
```
Open http://localhost:8081

Common configurations
---------------------
- Use a video file:
  ```bash
  # Place your file under ./videos and mount the folder (uncomment in compose if needed)
  VIDEO_SOURCE=/app/videos/your_clip.mp4 DEVICE=CPU docker compose up
  ```

- Use RTSP/HTTP stream:
  ```bash
  VIDEO_SOURCE=rtsp://user:pass@camera-ip/stream DEVICE=CPU docker compose up
  ```

- Linux webcam:
  ```yaml
  # In docker-compose.yml, uncomment:
  # devices:
  #   - "/dev/video0:/dev/video0"
  # privileged: true
  ```

Notes
-----
- Models are expected under `models/ov_models/` and are mounted into the container at `/app/models`.
- The app binds to `0.0.0.0:8081` in the container; port is published as `8081` on the host.
- On macOS/Windows, webcam access from containers is not supported by Docker Desktop; prefer a video file or network stream.
- If you see OpenVINO device issues, set `DEVICE=CPU`.

Troubleshooting
---------------
- Build errors about OpenCV GUI libs: this image installs runtime libs (libgl1, libglib2.0-0). If needed, ensure `opencv-python` is installed (already in requirements.txt).
- Large models: verify they are present in `./models` and readable in the container at `/app/models`.
- Port already in use: change the published port mapping in `docker-compose.yml` (e.g., `9090:8081`).
