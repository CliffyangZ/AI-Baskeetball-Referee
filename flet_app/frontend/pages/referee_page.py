"""
referee_page.py
--------------
Main referee interface page with real-time camera detection and statistics.
Integrates basketball, hoop, and pose detection with live statistics display.
"""

import flet as ft
import cv2
import base64
import asyncio
import numpy as np
from typing import Dict, Any
import threading
import time
import logging
import os

logger = logging.getLogger(__name__)

from backend.referee_integration import RefereeIntegration
from frontend.components.statistics_panel import StatisticsPanel
from backend.utils import get_scaled_size, get_windows_scale_factor

class RefereePage(ft.Column):
    """Main referee page with real-time detection and statistics"""
    
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.selected_camera = 0
        self.streaming = False
        self.cap = None
        self.referee_integration = RefereeIntegration()
        self.stats_update_task = None
        
        # Source selection: allow VIDEO_SOURCE env (index, file path, or URL)
        self.env_video_source = os.environ.get("VIDEO_SOURCE", "0")
        self._env_source_is_index = str(self.env_video_source).isdigit()
        if self._env_source_is_index:
            try:
                self.selected_camera = int(self.env_video_source)
            except Exception:
                self.selected_camera = 0
        
        # Window settings
        page.window_frameless = False
        page.window_full_screen = False
        page.window_maximized = True
        page.window_resizable = True
        page.title = "AI Basketball Referee - Real-time Detection"
        
        # Keyboard shortcuts
        def on_keyboard(e: ft.KeyboardEvent):
            if e.key == "Escape":
                self.stop_all()
                page.window_close()
            elif e.key == "Space":
                if self.streaming:
                    self.stop_camera()
                else:
                    self.start_camera(None)
        page.on_keyboard_event = on_keyboard
        
        # Initialize components
        self._setup_ui()
        
        # Initialize referee system
        self._initialize_referee()
        
        # Set up resize handler
        page.on_resized = self._on_resized

        self._on_resized(None)
    
    def _setup_ui(self):
        """Set up the user interface components"""
        
        # Camera dropdown
        use_index_source = self._env_source_is_index
        camera_options = self._get_available_cameras() if use_index_source else []
        self.camera_dropdown = ft.Dropdown(
            label="Select Camera",
            options=[ft.dropdown.Option(str(idx), f"Camera {idx}") for idx in camera_options],
            value=str(camera_options[0]) if (use_index_source and camera_options) else None,
            on_change=self._on_camera_change,
            width=150,
            disabled=not use_index_source
        )
        
        # Camera feed image
        self.camera_feed = ft.Image(
            src="assets/camera_placeholder.png", 
            expand=1,
            fit=ft.ImageFit.CONTAIN,
            border_radius=10,
            height=600  # Larger display height
        )
        
        # Status indicator
        self.status_text = ft.Text("Ready", size=14, color="#388E3C")
        self.detection_status = ft.Text("Detection: Stopped", size=14)
        
        # Control buttons
        self.start_btn = ft.ElevatedButton(
            "Start Detection", 
            icon="play_arrow", 
            on_click=self.start_camera,
            color="#FFFFFF",
            bgcolor="#388E3C"
        )
        self.stop_btn = ft.ElevatedButton(
            "Stop Detection", 
            icon="stop", 
            on_click=lambda e: self.stop_camera(),
            color="#FFFFFF",
            bgcolor="#D32F2F",
            disabled=True
        )
        
        # Statistics panel
        self.stats_panel = StatisticsPanel()
        
        # Navigation bar
        nav_bar = ft.Container(
            content=ft.Row([
                ft.Icon(name="sports_basketball", color="#F57C00", size=30),
                ft.Text("AI Basketball Referee", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(expand=1),  # Spacer
                self.status_text,
                ft.VerticalDivider(width=1),
                self.detection_status
            ], alignment=ft.MainAxisAlignment.START, spacing=10),
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            bgcolor="#E3F2FD",
            border=ft.border.only(bottom=ft.border.BorderSide(1, "#BDBDBD"))
        )
        
        # Source hint (visible when using non-index VIDEO_SOURCE)
        source_hint = (
            ft.Text(f"Using source: {self.env_video_source}", size=12, color="#757575")
            if not use_index_source else ft.Container()
        )
        
        # Camera section (left side)
        camera_section = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=self.camera_feed,
                    bgcolor="#F5F5F5",
                    border_radius=10,
                    expand=1
                ),
                ft.Container(
                    content=ft.Row([
                        self.camera_dropdown,
                        self.start_btn,
                        self.stop_btn,
                        ft.Container(expand=1),  # Spacer
                        source_hint,
                        ft.Text("Press SPACE to start/stop, ESC to exit", 
                               size=12, color="#757575")
                    ], spacing=10),
                    padding=10
                )
            ], spacing=10),
            expand=2,
            padding=10
        )
        
        # Right panel with statistics
        right_panel = ft.Column([
            self.stats_panel
        ], expand=1, scroll=ft.ScrollMode.AUTO)
        
        right_section = ft.Container(
            content=right_panel,
            expand=1,
            padding=10
        )
        
        # Main content area
        main_content = ft.Row([
            camera_section,
            ft.VerticalDivider(width=1),
            right_section
        ], expand=1, spacing=0)
        
        # Complete layout
        self.controls = [
            nav_bar,
            ft.Container(content=main_content, expand=1)
        ]
    
    def _initialize_referee(self):
        """Initialize the referee system"""
        try:
            # Try to initialize with default model paths
            success = self.referee_integration.initialize_referee()
            if success:
                self.status_text.value = "Referee System Ready"
                self.status_text.color = "#388E3C"
            else:
                self.status_text.value = "Referee System Failed"
                self.status_text.color = "#D32F2F"
        except Exception as e:
            self.status_text.value = f"Error: {str(e)[:30]}..."
            self.status_text.color = "#D32F2F"
        
        self.page.update()
    
    def _get_available_cameras(self, max_test=2):
        """Check for available camera indices"""
        available = []
        for i in range(max_test):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available.append(i)
                cap.release()
            except Exception as e:
                logger.warning(f"Error testing camera {i}: {e}")
                continue
        return available if available else [0]  # Default to 0 if none found
    
    def _on_camera_change(self, e):
        """Handle camera selection change"""
        self.selected_camera = int(e.control.value)
    
    
    def _on_resized(self, e):
        """Handle window resize"""
        if hasattr(self, 'camera_feed'):
            try:
                target_h = max(520, int(self.page.window_height * 0.7)) if self.page.window_height else 600
            except Exception:
                target_h = 600
            self.camera_feed.height = target_h
            self.page.update()
    
    def _stream_camera(self):
        """Main camera streaming loop with referee processing"""
        frame_count = 0
        last_fps_time = time.time()
        skip_frames = 0
        process_every_n_frames = 5  # Process every 5th frame for inference
        last_annotated_frame = None
        ui_update_counter = 0
        
        while self.streaming and self.cap and self.cap.isOpened():
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Preserve higher resolution for clarity; only downscale if extremely large
            height, width = frame.shape[:2]
            if width > 1280:  # Cap very large frames to 1280px width
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            display_frame = frame
            
            # Only process inference every N frames and don't wait for result
            if self.referee_integration.running and skip_frames % process_every_n_frames == 0:
                try:
                    self.referee_integration.process_frame_async(frame)
                except:
                    pass
            
            # Get annotated frame without blocking
            if self.referee_integration.running:
                try:
                    annotated_frame = self.referee_integration.get_annotated_frame()
                    if annotated_frame is not None:
                        last_annotated_frame = annotated_frame
                        display_frame = annotated_frame
                    elif last_annotated_frame is not None:
                        display_frame = last_annotated_frame
                except:
                    pass
            
            skip_frames += 1
            
            # High-quality encoding for clearer display
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Higher quality for clarity
            _, buffer = cv2.imencode(".jpg", frame_rgb, encode_param)
            img_data = base64.b64encode(buffer).decode("utf-8")
            self.camera_feed.src_base64 = img_data
            
            # Update FPS counter
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                self.detection_status.value = f"Detection: Running ({fps:.1f} FPS)"
                frame_count = 0
                last_fps_time = current_time
            
            # Update UI much less frequently
            ui_update_counter += 1
            if ui_update_counter >= 5:  # Update UI every 5th frame
                self.page.update()
                ui_update_counter = 0
            
            # Calculate remaining time for target FPS
            frame_time = time.time() - frame_start
            target_frame_time = 1/30  # Target 30 FPS
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
    
    def _update_statistics(self):
        """Update statistics display periodically"""
        while self.streaming:
            try:
                # Get latest statistics from referee
                stats = self.referee_integration.get_current_stats()
                
                # Update statistics panel
                self.stats_panel.update_statistics(stats)
                
                # Don't update page here - let camera thread handle UI updates
                time.sleep(0.5)  # Update every 500ms - much less frequent
            except Exception as e:
                print(f"Error updating statistics: {e}")
                time.sleep(2.0)
    
    def start_camera(self, e):
        """Start camera and detection"""
        if self.streaming:
            return
        
        # Open capture from env source (file/URL) or camera index
        source = self.selected_camera if self._env_source_is_index else self.env_video_source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.status_text.value = f"Error: Could not open source: {source}"
            self.status_text.color = "#D32F2F"
            self.page.update()
            return
        
        # Set camera properties for better clarity (camera may adjust to closest supported mode)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # Keep high capture FPS when supported
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
        # Start referee processing
        if not self.referee_integration.start_processing():
            self.status_text.value = "Error: Could not start referee"
            self.status_text.color = "#D32F2F"
            self.page.update()
            return
        
        # Update UI
        self.streaming = True
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_text.value = "Detection Running"
        self.status_text.color = "#1976D2"
        self.detection_status.value = "Detection: Starting..."
        
        # Start threads for camera streaming and statistics
        self.stream_thread = threading.Thread(target=self._stream_camera, daemon=True)
        self.stats_thread = threading.Thread(target=self._update_statistics, daemon=True)
        self.stream_thread.start()
        self.stats_thread.start()
        
        self.page.update()
    
    def stop_camera(self):
        """Stop camera and detection"""
        if not self.streaming:
            return
        
        # Stop streaming
        self.streaming = False
        
        # Wait for threads to finish
        try:
            if hasattr(self, "stream_thread") and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=0.5)
            if hasattr(self, "stats_thread") and self.stats_thread.is_alive():
                self.stats_thread.join(timeout=0.5)
        except Exception as e:
            logger.warning(f"Error joining threads: {e}")
        
        # Stop referee processing
        self.referee_integration.stop_processing()
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.status_text.value = "Detection Stopped"
        self.status_text.color = "#F57C00"
        self.detection_status.value = "Detection: Stopped"
        
        # Reset camera feed
        self.camera_feed.src = "assets/camera_placeholder.png"
        self.camera_feed.src_base64 = None
        
        self.page.update()
    
    def stop_all(self):
        """Stop all operations and cleanup"""
        self.stop_camera()
        # Additional cleanup if needed
