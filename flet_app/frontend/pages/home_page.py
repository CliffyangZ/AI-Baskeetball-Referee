"""
home_page.py
------------
Main screen layout for the AI Basketball Referee app.
Currently: Displays a title, start button, and shows a frame from webcam.
"""

from backend.utils import get_scaled_size, get_windows_scale_factor

import flet as ft
import cv2
import base64
import asyncio

class HomePage(ft.Column):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.selected_camera = 0
        self.streaming = False
        self.cap = None

        # # Initialize camera feed dimensions
        # self.section1_width = 0
        # self.section1_height = 0

        # Window settings       
        page.window_frameless = False        # Keep standard window frame
        page.window_full_screen = False      # Don't go absolute fullscreen
        page.window_maximized = True         # Fill the available desktop area
        page.window_resizable = False        # Disable resizing

        # Keyboard shortcut
        def on_keyboard(e: ft.KeyboardEvent):
            if e.key == "Escape":
                page.window_destroy()
        page.on_keyboard_event = on_keyboard

        # Camera dropdown
        camera_options = self._get_available_cameras()
        self.camera_dropdown = ft.Dropdown(
            label="Select Camera",
            options=[ft.dropdown.Option(str(idx)) for idx in camera_options],
            value=str(camera_options[0]) if camera_options else None,
            on_change=self._on_camera_change
        )

        # Camera feed image
        self.camera_feed = ft.Image(src="assets/camera_placeholder.png", expand=1)

        # Navigation bar
        nav_bar = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(name="settings"),
                    ft.Icon(name="leaderboard")
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=10,
            bgcolor="white",
            width=float("inf"), 
            height=60
        )

        # Sections
        self.section1 = ft.Container(
            content=self.camera_feed,
            bgcolor="grey200",
        )

        self.section2 = ft.Container(
            content=ft.Text("Section 2"),
            bgcolor="white",
            padding=10
        )

        self.section3 = ft.Container(
            content=ft.Row([
                self.camera_dropdown,
                ft.ElevatedButton("Start Camera", icon="play_arrow", on_click=self.start_camera),
                ft.ElevatedButton("Stop Camera", icon="stop", on_click=lambda e: self.stop_camera()),
                ft.Text("Decision: Waiting...", size=16)
            ], alignment=ft.MainAxisAlignment.START, spacing=10),
            bgcolor="white",
            padding=10
        )

        self.section4 = ft.Container(
            content=ft.Text("Section 4"),
            bgcolor="white",
            padding=10
        )

        self.top_row = ft.Row([self.section1, self.section2])
        self.bottom_row = ft.Row([self.section3, self.section4])

        main_layout = ft.Column([nav_bar, self.top_row, self.bottom_row], expand=1)
        self.controls = [main_layout]

        # Get Windows scaling factor
        self.scale_factor = get_windows_scale_factor()
        print(f"Windows scaling factor: {self.scale_factor}")

        page.on_resized = self._on_resized
        self._on_resized(None)
        

    def _get_available_cameras(self, max_test=5):
        """Check for available camera indices."""
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def _on_camera_change(self, e):
        """Update selected camera index."""
        self.selected_camera = int(e.control.value)

    def _on_resized(self, e):
        """Resize sections based on Section1's 2/3 width rule."""
        win_w = self.page.window.width
        win_h = self.page.window.height - 60  # minus navbar height

        # Section1: 2/3 width, 16:9 aspect ratio
        self.section1_width = int(win_w * (2/3))
        self.section1_height = int(self.section1_width * 9 / 16)
        
        # Cap height if needed
        if self.section1_height > win_h:
            self.section1_height = win_h
            self.section1_width = int(self.section1_height * 16 / 9)

        # Ensure height does not exceed 60% of window height
        max_height = int((win_h) * 0.6)
        if self.section1_height > max_height:
            self.section1_height = max_height
            # Recalculate width from height to keep 16:9 ratio
            self.section1_width = int(self.section1_height * 16 / 9)

        # Section1 container + camera feed
        self.section1.width = self.section1_width
        self.section1.height = self.section1_height
        self.camera_feed.width = self.section1_width
        self.camera_feed.height = self.section1_height

        # Section2: same height, fills remaining width
        self.section2.height = self.section1_height
        self.section2.width = win_w - self.section1_width

        # Section3: same width as section1, fills remaining height
        remaining_h = win_h - 60 - self.section1_height  # minus navbar height
        self.section3.width = self.section1_width
        self.section3.height = remaining_h

        # Section4: fills remaining height and width
        self.section4.width = self.section2.width
        self.section4.height = self.section3.height

        self.page.update()

        print(
            f"Resized to: {win_w}x{win_h}\n"
            f"Camera: {self.camera_feed.width}x{self.camera_feed.height}\n"
            f"Section1: {self.section1.width}x{self.section1.height}\n"
            f"Section2: {self.section2.width}x{self.section2.height}\n"
            f"Section3: {self.section3.width}x{self.section3.height}\n"
            f"Section4: {self.section4.width}x{self.section4.height}\n"
            f"Scale: {self.scale_factor}"
        )

    async def _stream_camera(self):
        while self.streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize to match section1 size
            if self.section1_width > 0 and self.section1_height > 0:
                frame = cv2.resize(frame, (self.section1_width, self.section1_height))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".png", frame_rgb)
            img_data = base64.b64encode(buffer).decode("utf-8")
            self.camera_feed.src_base64 = img_data
            self.page.update()
            await asyncio.sleep(1/30)  # ~30 FPS

    def start_camera(self, e):
        if self.streaming:
            return
        self.cap = cv2.VideoCapture(self.selected_camera)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        self.streaming = True
        asyncio.create_task(self._stream_camera())

    def stop_camera(self):
        if self.streaming:
            self.streaming = False
            if self.cap:
                self.cap.release()
                self.cap = None