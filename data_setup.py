import pandas as pd

def get_team_data(year):
    df = pd.read_csv(f'C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\data\\cbb{year}.csv')
    return df[['TEAM','CONF','G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']]

def get_bracket(year):
    df = pd.read_csv('C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\data\\data_cleaned.csv')
    df = df[df['YEAR'] == int(f'20{year}')]
    return df

def make_dataset(teams, bracket):
    bracket_a = bracket.copy()
    bracket_b = bracket.copy()
    bracket_a = bracket_a.rename({'WSEED':'A_SEED', 'WTEAM':'A_TEAM', 'LSEED':'B_SEED', 'LTEAM':'B_TEAM'}, axis=1)
    bracket_b = bracket_b.rename({'WSEED':'B_SEED', 'WTEAM':'B_TEAM', 'LSEED':'A_SEED', 'LTEAM':'A_TEAM'}, axis=1)
    bracket_a['RESULT'] = 0
    bracket_b['RESULT'] = 1
    
    bracket_df = pd.concat((bracket_a, bracket_b), axis=0)
    bracket_df = bracket_df[['YEAR', 'ROUND', 'A_SEED', 'A_TEAM', 'B_SEED', 'B_TEAM', 'RESULT']]
    
    temp_df = teams.copy()
    temp_df.columns = ['A_' + str(col) for col in temp_df.columns]
    a_df = pd.merge(bracket_df[['A_SEED' ,'A_TEAM']], temp_df, 'left', on='A_TEAM').drop(['A_TEAM', 'A_CONF'], axis=1)

    temp_df = teams.copy()
    temp_df.columns = ['B_' + str(col) for col in temp_df.columns]
    b_df = pd.merge(bracket_df[['B_SEED' ,'B_TEAM']], temp_df, 'left', on='B_TEAM').drop(['B_TEAM', 'B_CONF'], axis=1)
    
    df = pd.concat((a_df, b_df, bracket_df.RESULT.reset_index(drop=True)), axis=1)
    # y = bracket_df.RESULT
    return df

def fix_23():
    val = pd.read_csv(f'C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\data\\cbb23copy.csv', header=None)
    val = pd.Series(val.values.reshape(-1)).dropna()
    val = val[~val.str.contains('vs.')][22:]
    val = pd.DataFrame(val.values.reshape((-1, 40)))
    print(val)

    val.columns = [
        'Rk','TEAM','CONF','G',
        'W-L', 'Conw-l',
        'ADJOE', '_rank',
        'ADJDE', '_rank',
        'BARTHAG', '_rank',
        'EFG_O', '_rank',
        'EFG_D', '_rank',
        'TOR', '_rank',
        'TORD', '_rank',
        'ORB', '_rank',
        'DRB', '_rank',
        'FTR', '_rank',
        'FTRD', '_rank',
        '2P_O', '_rank',
        '2P_D', '_rank',
        '3P_O', '_rank',
        '3P_D', '_rank',
        'ADJ_T', '_rank',
        'WAB', '_rank'
    ]
    def get_wins(s):
        return s.split('-')[0]
    val['W'] = val['W-L'].apply(get_wins)
    print(val)
    val = val[['TEAM','CONF','G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']]
    print(val)
    val.to_csv('C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\data\\cbb23.csv')

if __name__ == '__main__':
    total_df = pd.DataFrame()

    for i in range(9):
        year = i + 13
        print(f'Year: 20{year}')

        team_df = get_team_data(year)
        bracket = get_bracket(year)

        df = make_dataset(team_df, bracket)
        df['YEAR'] = f'20{year}'
        total_df = pd.concat((df, total_df), axis=0)
        # print(total_df)

    total_df = total_df.reset_index(drop=True)
    total_df.to_csv('C:\\Users\\vauda\\Documents\\work\\PS\\NCRMadness\\data\\training_data.csv', index=False)

    print(get_team_data(23))
        # print(team_df)

        # print(bracket)
        # print(all_teams)
        # print(team_seeds[~team_seeds['TEAM'].isin(team_df['TEAM'])]['TEAM'].sort_values())
        # # print(team_df[~team_df['TEAM'].isin(team_seeds['TEAM'])]['TEAM'].sort_values())
        # print(team_seeds['TEAM'].isin(team_df['TEAM']).all())


    