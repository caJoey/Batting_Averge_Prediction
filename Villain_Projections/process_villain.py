# scrape and create the villain_guess.csv file
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

# grab villain predictions
STEAMER_URL = 'https://web.archive.org/web/20250326192222/https://www.fangraphs.com/projections'
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
df_guess = pd.DataFrame(stats_list)[['AVG', 'xMLBAMID']]
df_guess = df_guess.dropna()
df_guess = df_guess.rename(columns={'xMLBAMID': 'key_mlbam', 'AVG': 'BA_guess'})
df_guess = df_guess.dropna()
df_guess['key_mlbam'] = df_guess['key_mlbam'].astype('int64')
df_guess.to_csv('villain_guess.csv', index=False)
