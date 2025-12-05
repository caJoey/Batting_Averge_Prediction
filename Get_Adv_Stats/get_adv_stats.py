'''
For getting statcast stats and storing in adv_stats.csv
 - No webscraping necessary, only requests package hopefully
Example link:
 - Grabs advanced stats in the "batted ball" profile page for players from dateStart to dateEnd and downloads the associated csv
https://baseballsavant.mlb.com/leaderboard/batted-ball?dateStart=2025-07-20&dateEnd=2025-09-15&csv=true
Add that minimum swings for a player must be 5:
https://baseballsavant.mlb.com/leaderboard/batted-ball?dateStart=2025-07-20&dateEnd=2025-09-15&minSwings=5&csv=true

'''

import pandas as pd
import requests
from io import StringIO

# 1 instance of these per category of statcast stats
class Adv_Stats:
    def __init__(self, name, link, columns, mini):
        self.name = name
        self.link = link
        self.columns = columns
        # mini is extra constraint in URL that expands the # of qualified batters
        self.mini = mini

# grab adv stats from 2015-2025 given Adv_Stats object
def grab_adv_stats(adv_stats):
    name, link, columns, mini = \
    adv_stats.name, adv_stats.link, adv_stats.columns, adv_stats.mini
    year_dfs = []
    for year in range(2015, 2026):
        # inconsistent
        if name != 'Batting_Run_Value':
            response = requests.get(f'{link}?type=batter&year={year}{mini}&csv=true')
        else:
            response = requests.get(f'{link}?group=Batter&year={year}{mini}&csv=true')
        csv_content = response.text
        # makes it behave like a text file open in memory and then we can read it using pd
        # print(csv_content[:1000])
        df = pd.read_csv(StringIO(csv_content))
        df = df[columns]
        df['year'] = year
        year_dfs.append(df)
    returner = pd.concat(year_dfs)
    # Batted_Ball has 'id' instead of 'player_id'
    if name == 'Batted_Ball':
        returner = returner.rename(columns={'id': 'player_id'})
    return returner

# holds Adv_Stats object for each statcast category
stack = []
# TODO: normalize id columns
# Expected Stats
stack.append(Adv_Stats('Expected_Stats',
'https://baseballsavant.mlb.com/leaderboard/expected_statistics',
["player_id","year","bip","est_ba","est_ba_minus_ba_diff","est_slg","est_slg_minus_slg_diff","woba","est_woba","est_woba_minus_woba_diff"],
'&filterType=pa&min=100'))
# Batted Ball
stack.append(Adv_Stats('Batted_Ball',
'https://baseballsavant.mlb.com/leaderboard/batted-ball',
["id","bbe","gb_rate","air_rate","fb_rate","ld_rate","pu_rate","pull_rate","straight_rate","oppo_rate","pull_gb_rate","straight_gb_rate","oppo_gb_rate","pull_air_rate","straight_air_rate","oppo_air_rate"],
'&minSwings=100'))
# Sprint Speed
stack.append(Adv_Stats('Sprint_Speed',
'https://baseballsavant.mlb.com/leaderboard/sprint_speed',
["player_id","position","competitive_runs","sprint_speed"],
''))
# Exit Velo & Barrels 
stack.append(Adv_Stats('Exit_Velo_Barrels',
'https://baseballsavant.mlb.com/leaderboard/statcast',
["player_id","attempts","avg_hit_angle","anglesweetspotpercent","max_hit_speed","avg_hit_speed","ev50","fbld","gb","max_distance","avg_distance","avg_hr_distance","ev95plus","ev95percent","barrels","brl_percent","brl_pa"],
'&min=100'))
# Batting Run Value 
stack.append(Adv_Stats('Batting_Run_Value',
'https://baseballsavant.mlb.com/leaderboard/swing-take',
["year","player_id","team_id","pitches","runs_all","runs_heart","runs_shadow","runs_chase","runs_waste"],
'&min=100'))

for i, adv_stats in enumerate(stack):
    # replace Adv_Stats object with associated df
    stack[i] = grab_adv_stats(adv_stats)
# merge advanced stats together
df = stack.pop()
# inner join ON [player_id, year]
while stack:
    df = pd.merge(df, stack.pop())
df = df.rename(columns={'player_id': 'key_mlbam'})
df['position'] = df['position'].fillna('UNK')
# positions need to be one hot encoded
position_dummies = pd.get_dummies(df['position'], prefix='POS')
for col in position_dummies.columns:
    if position_dummies[col].dtype == 'bool':
        position_dummies[col] = position_dummies[col].astype(int)
df = pd.concat([df, position_dummies], axis=1)
df = df.drop('position', axis=1)
# team_ids also need to be one hot encoded
team_dummies = pd.get_dummies(df['team_id'], prefix='TEAM')
for col in team_dummies.columns:
    if team_dummies[col].dtype == 'bool':
        team_dummies[col] = team_dummies[col].astype(int)
df = pd.concat([df, team_dummies], axis=1)
df = df.drop('team_id', axis=1)
print(df.head())
print(f'row len {len(df)}')
print(f'col len {len(df.columns)}')
df.to_csv('adv_stats.csv', index=False)
