import pandas as pd

letters = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
# we want only key_mlbam (statcast) and key_bbref (baseball reference, normal data)
good_cols = ['key_bbref', 'key_fangraphs']
big_df = pd.DataFrame()
# process the csvs
for let in letters:
    csv_path = f'register/data/people-{let}.csv'
    try:
        df = pd.read_csv(csv_path)
        df = df[good_cols]
        df = df.dropna()
        big_df = pd.concat([big_df, df])
    except Exception as e:
        print(f"Error processing {let}: {e}. Skipping this file.")
big_df['key_fangraphs'] = big_df['key_fangraphs'].astype('int64')
# replace or create new file to copy csv to
big_df.to_csv('mapping_fan.csv', index=False)
