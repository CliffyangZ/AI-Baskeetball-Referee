class Scoreboard:
    def __init__(self):
        self.team_a_score = 0
        self.team_b_score = 0
        self.game_status = "Not Started"

    def update_score(self, team, points):
        if team == "A":
            self.team_a_score += points
        elif team == "B":
            self.team_b_score += points

    def set_game_status(self, status):
        self.game_status = status

    def display(self):
        return f"Score: Team A - {self.team_a_score}, Team B - {self.team_b_score}, Status - {self.game_status}"