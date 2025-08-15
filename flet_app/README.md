# Program Structure

```
ai_basketball_referee/
│
├── main.py                  # Entry point for your Flet app
│
├── frontend/                # All UI-related code
│   ├── __init__.py
│   ├── pages/                # Different screens/pages
│   │   ├── home_page.py
│   │   ├── settings_page.py
│   │   └── results_page.py
│   │
│   ├── components/           # Reusable UI widgets
│   │   ├── camera_view.py
│   │   ├── scoreboard.py
│   │   └── decision_card.py
│   │
│   └── styles.py             # Colors, fonts, layout settings
│
├── backend/                  # AI + processing logic
│   ├── __init__.py
│   ├── inference.py          # OpenVINO model loading + inference
│   ├── video_capture.py      # Webcam / video stream handling
│   └── utils.py              # Helper functions
│
├── models/                   # OpenVINO exported model files
│   ├── model.xml
│   └── model.bin
│
├── assets/                   # Images, icons, sample videos
│   ├── logo.png
│   └── court_background.jpg
│
└── requirements.txt          # Python dependencies
```