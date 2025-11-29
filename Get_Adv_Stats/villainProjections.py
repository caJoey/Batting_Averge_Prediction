'''
Projections of competing models
 - Steamer
'''

import requests
from bs4 import BeautifulSoup
import json

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
json_string = script_tag.string.strip()
json_data = json.loads(json_string)
stats_list = json_data['props']['pageProps']['dehydratedState']['queries'][0]['state']['data']
print(stats_list[0])

# with open('soup.html', 'w', encoding='utf-8') as f:
#     f.write(soup.prettify())
