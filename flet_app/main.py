"""
main.py
--------
Entry point for the AI Basketball Referee desktop app.
Uses Flet for the frontend UI and OpenVINO for backend AI inference.
Checks if running inside a virtual environment for dependency isolation.
"""

import os
import sys
import flet as ft
from frontend.pages.home_page import HomePage
from backend.utils import get_scaled_size, make_aspect_ratio_handler

def check_virtualenv():
    """
    Checks if the script is running inside a virtual environment.
    If not, prints a warning message and exits the program.
    """
    if sys.prefix == sys.base_prefix:
        print(
            "\n⚠️ You are not running inside a virtual environment (venv)."
            "\n💡 It's recommended to activate your venv before running this app."
            "\nExample (Windows): venv\\Scripts\\activate"
            "\nExample (Linux/Mac): source venv/bin/activate\n"
        )
        sys.exit(1)  # Exit if not in venv

def main(page: ft.Page):
    """
    Main function that initializes the Flet app.
    """

    # ----- PAGE SETTINGS -----
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 10
    page.theme_mode = ft.ThemeMode.DARK

    # ----- ENFORCE ASPECT RATIO -----
    page.on_window_event = make_aspect_ratio_handler(16/9)
    page.update()

    # ----- LOAD HOME PAGE -----
    home = HomePage(page)
    page.add(home)  # Add home screen to the main page

if __name__ == "__main__":
    check_virtualenv()
    ft.app(target=main)