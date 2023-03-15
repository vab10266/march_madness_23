import pandas as pd
from NCRMadness.game_model import column_selector

class Game:
    def __init__(self, a_team=None, b_team=None, winner=None):
        self.a_team = a_team
        self.b_team = b_team
        self.winner = winner

class Round:
    def __init__(self, games, winners=None):
        self.games = games
        self.winners = winners
    
    def get_X_data(self, team_df, cols):
        # generate the X data for the model to predict the winners.
        pass

    def get_winners(self, y, rand=False):
        # given the model's predictions, set and return the winning teams.
        # if rand, a_team is chosen as the winner with a likelihood of p
        # else, use a threshold of 0.5
        pass

class Bracket:
    def __init__(self, teams, team_df, rounds=None):
        self.teams = teams
        self.team_df = team_df
        self.rounds = rounds

    def generate_winners(self, model, cols, rand=False):
        # given the initial conditions and a model, predict the outcome
        pass
    
    def eval(self, true):
        # Assuming that true is the true bracket, return the score of self.
        pass

    def from_csv(filename):
        # Initialize from stored csv
        pass

    def from_binary(bin, teams):
        # given a binary string of len(32 + 16 + ... + 1) and team names in order
        # return a bracket object resulting from those results
        pass

if __name__ == '__main__':
    pass
