"""
statistics_panel.py
------------------
Real-time statistics display panel for basketball referee system.
Shows shot statistics, dribble count, step count, and violations.
"""

import flet as ft
from typing import Dict, Any

class StatisticsPanel(ft.Container):
    """Real-time statistics display panel"""
    
    def __init__(self):
        super().__init__()
        
        # Statistics text controls
        self.shot_stats_text = ft.Text("Shots: 0/0 (0.0%)", size=18, weight=ft.FontWeight.BOLD)
        self.dribble_text = ft.Text("Dribbles: 0", size=16)
        self.steps_text = ft.Text("Total Steps: 0", size=16)
        self.violations_text = ft.Text("Violations: 0", size=16)
        self.fps_text = ft.Text("FPS: 0.0", size=14)
        self.processing_text = ft.Text("Processing: 0.0ms", size=14)
        
        # Player step details
        self.player_steps_column = ft.Column(spacing=5)
        
        # Violations list
        self.violations_column = ft.Column(spacing=3)
        
        # Performance metrics
        performance_row = ft.Row([
            self.fps_text,
            ft.VerticalDivider(width=1),
            self.processing_text
        ], spacing=10)
        
        # Main statistics column
        stats_column = ft.Column([
            ft.Text("Basketball Statistics", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700),
            ft.Divider(height=1),
            self.shot_stats_text,
            self.dribble_text,
            self.steps_text,
            self.player_steps_column,
            ft.Divider(height=1),
            self.violations_text,
            self.violations_column,
            ft.Divider(height=1),
            ft.Text("Performance", size=16, weight=ft.FontWeight.BOLD),
            performance_row
        ], spacing=8)
        
        # Container setup
        self.content = ft.Container(
            content=stats_column,
            padding=15,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            border=ft.border.all(1, ft.Colors.GREY_300)
        )
        
    def update_statistics(self, stats: Dict[str, Any]):
        """Update the statistics display with new data"""
        
        # Shot statistics
        shot_makes = stats.get('shot_makes', 0)
        shot_attempts = stats.get('shot_attempts', 0)
        shooting_percentage = stats.get('shooting_percentage', 0.0)
        self.shot_stats_text.value = f"Shots: {shot_makes}/{shot_attempts} ({shooting_percentage:.1f}%)"
        
        # Set color based on shooting percentage
        if shooting_percentage >= 50:
            self.shot_stats_text.color = ft.Colors.GREEN_700
        elif shooting_percentage >= 30:
            self.shot_stats_text.color = ft.Colors.ORANGE_700
        else:
            self.shot_stats_text.color = ft.Colors.RED_700
        
        # Dribble count
        dribble_count = stats.get('dribble_count', 0)
        self.dribble_text.value = f"Dribbles: {dribble_count}"
        
        # Total steps
        total_steps = stats.get('total_steps', 0)
        self.steps_text.value = f"Total Steps: {total_steps}"
        
        # Individual player steps
        step_count = stats.get('step_count', {})
        self.player_steps_column.controls.clear()
        for player_id, steps in step_count.items():
            player_text = ft.Text(f"  Player {player_id}: {steps} steps", size=14, color=ft.Colors.GREY_700)
            self.player_steps_column.controls.append(player_text)
        
        # Violations
        violations = stats.get('violations', [])
        violation_count = len(violations)
        self.violations_text.value = f"Violations: {violation_count}"
        
        # Set color based on violation count
        if violation_count == 0:
            self.violations_text.color = ft.Colors.GREEN_700
        elif violation_count <= 2:
            self.violations_text.color = ft.Colors.ORANGE_700
        else:
            self.violations_text.color = ft.Colors.RED_700
        
        # Violations list
        self.violations_column.controls.clear()
        for i, violation in enumerate(violations[-5:]):  # Show last 5 violations
            violation_type = violation.get('type', 'Unknown')
            violation_text = ft.Text(f"  {i+1}. {violation_type.upper()}", 
                                   size=12, color=ft.Colors.RED_600)
            self.violations_column.controls.append(violation_text)
        
        # Performance metrics
        fps = stats.get('fps', 0.0)
        processing_time = stats.get('processing_time', 0.0)
        self.fps_text.value = f"FPS: {fps:.1f}"
        self.processing_text.value = f"Processing: {processing_time:.1f}ms"
        
        # Set FPS color
        if fps >= 25:
            self.fps_text.color = ft.Colors.GREEN_700
        elif fps >= 15:
            self.fps_text.color = ft.Colors.ORANGE_700
        else:
            self.fps_text.color = ft.Colors.RED_700


