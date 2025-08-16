"""
main_referee.py
--------------
Main entry point for the AI Basketball Referee with real-time detection interface.
Launches the referee page with integrated statistics and camera detection.
"""

import flet as ft
from frontend.pages.referee_page import RefereePage

def main(page: ft.Page):
    """Main function for the referee application"""
    
    # Page settings
    page.title = "AI Basketball Referee - Real-time Detection"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0
    
    # Create and add referee page
    referee_page = RefereePage(page)
    page.add(referee_page)
    
    # Initial update
    page.update()

if __name__ == "__main__":
    ft.app(target=main, port=8081, view=ft.AppView.WEB_BROWSER)
