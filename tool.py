import pickle
import pandas as pd
from data_setup import get_team_data

def generate_round(teams):
    a_teams = teams.copy()
    b_teams = teams.copy()
    a_teams = a_teams.iloc[::2].reset_index(drop=True)
    b_teams = b_teams.iloc[1::2].reset_index(drop=True)
    a_teams = a_teams.drop('TEAM', axis=1)
    b_teams = b_teams.drop('TEAM', axis=1)
    a_teams = a_teams.drop('CONF', axis=1)
    b_teams = b_teams.drop('CONF', axis=1)
    a_teams = a_teams.add_prefix('A_')
    b_teams = b_teams.add_prefix('B_')
    return pd.concat((a_teams, b_teams), axis=1)

def predict_round(teams, model, cols, prob=False):
    # print(teams)
    X = generate_round(teams)
    # print(X)
    
    x_cols = []
    for col in cols:
        x_cols.append(X.columns[col])
    for col in cols:
        x_cols.append(X.columns[20+col])

    X = X[x_cols]
    # print(X)
    print(model.predict_proba(X))
    if prob == True:
        return model.predict_proba(X)
    return model.predict(X)

def get_winning_teams(teams, preds):
    # print(teams)
    print(preds)
    winners = pd.DataFrame()
    ind = 0
    # print(teams.shape)
    for i in range(preds.shape[0]):
        # print(ind, preds[i])
        winners = pd.concat((winners, teams.iloc[ind + preds[i]]), axis=1)
        ind += 2
    return winners.T

if __name__ == '__main__':
    
    team_data = get_team_data(23)
    df = pd.read_csv('C:\\Users\\vauda\\Documents\\work\\PS\\march_madness_23\\data\\teams23.csv')
    # print(df.columns)
    model = pickle.load(open('C:\\Users\\vauda\\Documents\\work\\PS\\march_madness_23\\game_model.sav', 'rb'))
    cols = pickle.load(open('C:\\Users\\vauda\\Documents\\work\\PS\\march_madness_23\\cols.sav', 'rb'))#[3, 4, 5, 6, 7, 10, 11] [1, 3, 4, 11]
    teams_23 = pd.merge(df, team_data, 'left', on='TEAM')
    print(teams_23)
    print('='*100)
    r1_preds = predict_round(teams_23, model, cols)
    r1_preds[18] = 1
    r1_preds[14] = 0
    r2_teams = get_winning_teams(teams_23, r1_preds)
    print(r2_teams)
    print('='*100)
    r2_preds = predict_round(r2_teams, model, cols)
    r3_teams = get_winning_teams(r2_teams, r2_preds)
    print(r3_teams)
    print('='*100)
    r3_preds = predict_round(r3_teams, model, cols)
    r3_preds[0] = 0
    r3_preds[6] = 0
    r4_teams = get_winning_teams(r3_teams, r3_preds)
    print(r4_teams)
    print('='*100)
    r4_preds = predict_round(r4_teams, model, cols)
    r4_preds[1] = 1
    r5_teams = get_winning_teams(r4_teams, r4_preds)
    print(r5_teams)
    print('='*100)
    r5_preds = predict_round(r5_teams, model, cols)
    r6_teams = get_winning_teams(r5_teams, r5_preds)
    print(r6_teams)
    print('='*100)
    r6_preds = predict_round(r6_teams, model, cols)
    r7_teams = get_winning_teams(r6_teams, r6_preds)
    print(r7_teams)