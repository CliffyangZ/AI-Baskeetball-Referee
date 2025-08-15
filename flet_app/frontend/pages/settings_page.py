# settings_page.py

from flet import Page, Column, TextField, ElevatedButton, Text

def settings_page(page: Page):
    page.title = "Settings"
    page.vertical_alignment = "start"

    settings_column = Column()

    # Add a text field for user preferences
    preference_input = TextField(label="Enter your preference", width=300)
    settings_column.controls.append(preference_input)

    # Add a button to save settings
    save_button = ElevatedButton(text="Save Settings", on_click=lambda e: save_settings(preference_input.value))
    settings_column.controls.append(save_button)

    # Add a label for feedback
    feedback_label = Text("")
    settings_column.controls.append(feedback_label)

    page.add(settings_column)

def save_settings(preference):
    # Logic to save user preferences can be implemented here
    print(f"Settings saved: {preference}")  # Placeholder for actual save logic
    feedback_label.value = "Settings saved successfully!"