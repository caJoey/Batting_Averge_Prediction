'''
Produce projections of competing models
in villain_guess.csv
 - Steamer
'''

import pandas as pd
import os

STATS_PATH = '../Get_Stats/stats.csv'
MAP_PATH = '../ID_Mapping/mapping.csv'
GUESS_PATH = 'villain_guess.csv'
if not os.path.exists(STATS_PATH):
    print('Error: Run \'node getStats.js\' in Get_Stats folder first!')
    exit()
if not os.path.exists(MAP_PATH):
    print('Error: Run \'process_people.py\' in ID_Mapping folder first!')
    exit()
if not os.path.exists(GUESS_PATH):
    print('Error: Run \'process_villain.py\' in Villain_Projections folder first!')
    exit()
df_guess = pd.read_csv(GUESS_PATH)
# grab mapping DF and actual 2025 stats
df_actual = pd.read_csv(STATS_PATH)
# remove 2025 rookies (because our model uses stats from previous year and disallows rookies)
season_counts = df_actual.groupby('key_bbref')['year'].count()
non_rookies = season_counts[season_counts > 1].index
df_actual = df_actual[df_actual['key_bbref'].isin(non_rookies)]
# df_actual = df_actual[df_actual.groupby('key_bbref').transform('count') > 1]
df_actual = df_actual[df_actual['year'] == 2025]
df_actual = df_actual[['BA', 'key_bbref']]
df_map = pd.read_csv(MAP_PATH)
actual_map = pd.merge(df_actual, df_map)
map_guess = pd.merge(df_map, df_guess)
actual_guess = pd.merge(actual_map, map_guess)
abs_error = (actual_guess['BA_guess'] - actual_guess['BA']).abs()
print(f'abs_error: {abs_error.mean()}')
# print('actual_guess')
# print(actual_guess.head())
# print(f'len(actual_guess) {len(actual_guess)}')
