# File: /ai_basketball_referee/ai_basketball_referee/frontend/pages/results_page.py

from flet import Page, Column, Text, Container

def results_page(page: Page):
    page.title = "Results"
    
    # Create a container for the results
    results_container = Container(
        content=Column(
            [
                Text("AI Referee Results", size=24, weight="bold"),
                Text("Statistics and Outcomes", size=18),
                # Placeholder for results data
                Text("Game Duration: 00:00:00", size=16),
                Text("Total Fouls: 0", size=16),
                Text("Decisions Made: 0", size=16),
            ]
        ),
        padding=20,
        alignment="center"
    )
    
    page.add(results_container)