# assumes stats.csv already exists

import sys
import pandas as pd
import io

year = sys.argv[1]

# cols tp keep
cols = [
    'Player','Player-additional', 'Age', 'WAR', 'G', 'PA', 'AB', 'H', '2B', '3B', 'HR',
    'RBI', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'rOBA', 'Rbat+', 'HBP', 'SH', 'SF', 'IBB'
]
# just for copypaste to csv
header = 'player,key_bbref,Age,WAR,G,PA,AB,H,2B,3B,HR,RBI,SO,BA,OBP,SLG,OPS,OPS+,rOBA,Rbat+,HBP,SH,SF,IBB,year'

csv_text = sys.stdin.read()
# replace unknown chars with '�' - prevents issues
csv_text = csv_text.encode('utf-8', errors='replace').decode('utf-8')

df = pd.read_csv(io.StringIO(csv_text))
df = df[cols]
# preprocess 1: delete < 100 PA
df = df[df['PA'] >= 100]
# 2: merge players with same id
idx_max_g = df.groupby('Player-additional')['G'].idxmax()
df = df.loc[idx_max_g]
# 3: append year column
df['year'] = year
# Appends, no header, no left column
df = df.rename(columns={'id': 'key_bbref'})
df.to_csv('stats.csv', mode='a', header=False, index=False)

# gets here if successful
# keep track of successful years in a text file
with open('years.txt', 'a') as f:
    f.write(f"{str(year)},")
