# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Avoid Python writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for OpenCV and media handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir and copy requirements
WORKDIR /app
COPY requirements.txt ./

# Install Python dependencies (use root requirements to cover AI_referee/tracker)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Expose Flet web port
EXPOSE 8081

# Default runtime configuration
ENV VIDEO_SOURCE=0 \
    DEVICE=AUTO \
    RULES=FIBA

# Run the Flet referee UI. We bind to 0.0.0.0 without changing source code.
# Workdir is flet_app so imports like `from frontend.pages...` continue to work.
WORKDIR /app/flet_app
CMD ["python", "-c", "import flet as ft; from main_referee import main; ft.app(target=main, host='0.0.0.0', port=8081, view=ft.AppView.WEB_BROWSER)"]
