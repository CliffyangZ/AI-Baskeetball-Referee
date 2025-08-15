from flet import Column, Container, Text, Card, Row

def decision_card(decision_text):
    return Card(
        content=Column(
            [
                Text("AI Decision", size=20, weight="bold"),
                Row(
                    [
                        Container(
                            content=Text(decision_text, size=16),
                            padding=10,
                        )
                    ],
                    alignment="center"
                )
            ],
            alignment="center",
            spacing=10
        ),
        elevation=2,
        padding=10,
        bgcolor="lightblue"
    )