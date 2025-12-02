'''
Producerojections of competing models
in villain_guess.csv
 - Steamer
'''

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os

# grab villain predictions
STEAMER_URL = 'https://web.archive.org/web/20250326192222/https://www.fangraphs.com/projections'
response = requests.get(STEAMER_URL)
if response.status_code == 200:
    # Proceed if the request was successful
    pass
else:
    print(f"Error: Could not retrieve data. Status code: {response.status_code}")
    exit()
soup = BeautifulSoup(response.text, 'html.parser')
script_tag = soup.find('script', id='__NEXT_DATA__')
if not script_tag:
    print('Error: Could not find the __NEXT_DATA__ script tag.')
    exit()
STATS_PATH = '../Get_Stats/stats.csv'
MAP_PATH = '../ID_Mapping/mapping.csv'
if not os.path.exists(STATS_PATH):
    print('Error: Run \'node getStats.js\' in Get_Stats folder first!')
    exit()
if not os.path.exists(MAP_PATH):
    print('Error: Run \'process_people.py\' in ID_Mapping folder first! ')
    exit()
json_string = script_tag.string.strip()
json_data = json.loads(json_string)
stats_list = json_data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
df_guess = pd.DataFrame(stats_list)[['AVG', 'xMLBAMID']]
df_guess = df_guess.dropna()
df_guess = df_guess.rename(columns={'xMLBAMID': 'key_mlbam', 'AVG': 'BA_guess'})
df_guess = df_guess.dropna()
df_guess['key_mlbam'] = df_guess['key_mlbam'].astype('int64')
# grab mapping DF and actual 2025 stats
df_actual = pd.read_csv(STATS_PATH)
print(df_actual.head())
# remove 2025 rookies (because our model uses stats from previous year and disallows rookies)
season_counts = df_actual.groupby('key_bbref')['year'].count()
non_rookies = season_counts[season_counts > 1].index
df_actual = df_actual[df_actual['key_bbref'].isin(non_rookies)]
# df_actual = df_actual[df_actual.groupby('key_bbref').transform('count') > 1]
df_actual = df_actual[df_actual['year'] == 2025]
print(len(df_actual))
print('-----')
df_actual = df_actual[['BA', 'key_bbref']]
df_map = pd.read_csv(MAP_PATH)
actual_map = pd.merge(df_actual, df_map)
map_guess = pd.merge(df_map, df_guess)
actual_guess = pd.merge(actual_map, map_guess)
abs_error = (actual_guess['BA_guess'] - actual_guess['BA']).abs()
print(f'abs_error: {abs_error.mean()}')
print('actual_guess')
print(actual_guess.head())
print(f'len(actual_guess) {len(actual_guess)}')

print('good, continue')
