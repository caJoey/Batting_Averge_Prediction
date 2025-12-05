# scrape and create the villain_guess.csv file
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

# grab villain predictions
STEAMER_URL = 'https://web.archive.org/web/20240204123251/https://www.fangraphs.com/projections?type=thebatx'
response = requests.get(STEAMER_URL)
if response.status_code != 200:
    print(f"Error: Could not request data. Status code: {response.status_code}")
    exit()
soup = BeautifulSoup(response.text, 'html.parser')
script_tag = soup.find('script', id='__NEXT_DATA__')
if not script_tag:
    print('Error: Could not find the __NEXT_DATA__ script tag.')
    exit()
json_string = script_tag.string.strip()
json_data = json.loads(json_string)
stats_list = json_data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
# df_guess = pd.DataFrame(stats_list)[['AVG', 'xMLBAMID']]
df_guess = pd.DataFrame(stats_list)[['AVG', 'playerid']]
df_guess = df_guess.dropna()
print(df_guess.head())
# df_guess = df_guess.rename(columns={'xMLBAMID': 'key_mlbam', 'AVG': 'BA_guess'})
df_guess = df_guess.rename(columns={'playerid': 'key_fangraphs', 'AVG': 'BA_guess'})
print(f'pre filter rows {len(df_guess)}')
# some columns arent numbers
df_guess['col1_numeric'] = pd.to_numeric(df_guess['key_fangraphs'], errors='coerce')
df_guess = df_guess[df_guess['col1_numeric'].notna()]
df_guess = df_guess.drop(columns=['col1_numeric'])
df_guess['key_fangraphs'] = df_guess['key_fangraphs'].astype('int64')
print(f'post filter rows {len(df_guess)}')
df_guess.to_csv('villain_guess_batx.csv', index=False)
