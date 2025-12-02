'''
For getting statcast stats
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

CSV_URL = 'https://baseballsavant.mlb.com/leaderboard/batted-ball?dateStart=2025-07-20&dateEnd=2025-09-15&csv=true'

response = requests.get(CSV_URL)
if response.status_code == 200:
    # Proceed if the request was successful
    pass
else:
    print(f"Error: Could not retrieve data. Status code: {response.status_code}")
    exit()
csv_content = response.text
# makes it behave like a text file open in memory and then we can read it using pd
df = pd.read_csv(StringIO(csv_content))
print(df.head())
print(len(df))
