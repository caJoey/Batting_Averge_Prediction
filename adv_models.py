from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt

STATS_PATH = './Get_Stats/stats.csv'
MAP_PATH = './ID_Mapping/mapping.csv'
ADV_STATS_PATH = './Get_Adv_Stats/adv_stats.csv'
if not os.path.exists(STATS_PATH):
    print('Error: Run \'node getStats.js\' in Get_Stats folder first!')
    exit()
if not os.path.exists(MAP_PATH):
    print('Error: Run \'process_people.py\' in ID_Mapping folder first!')
    exit()
if not os.path.exists(ADV_STATS_PATH):
    print('Error: Run \'get_adv_stats.py\' in Get_Adv_Stats folder first!')
    exit()
df_stats = pd.read_csv(STATS_PATH)
df_map = pd.read_csv(MAP_PATH)
df_adv_stats = pd.read_csv(ADV_STATS_PATH)
# inner join on key_bbref
stats_map = pd.merge(df_stats, df_map)
print(f'basic row len {len(df_stats)}')
print(f'basic col len {len(df_stats.columns)}\n')
# inner join on key_mlbam
map_adv = pd.merge(df_map, df_adv_stats)
print(f'adv row len {len(df_adv_stats)}')
print(f'adv col len {len(df_adv_stats.columns)}\n')
# inner join on [key_bbref, key_mlbam]
# Update your merge to include 'year'
df = pd.merge(stats_map, map_adv)

# drop stuff we don't care about for machine learning
df = df.drop(['player', 'key_mlbam'], axis=1)
df = df.sort_values(by=['key_bbref', 'year']).reset_index(drop=True)

# feature engineering
# strikout rate
df['K%'] = df['SO'] / df['PA']
# bases on walk / hbp percentage
df['BB%'] = (df['BB'] + df['HBP']) / df['PA']
# isolated power
df['ISO'] = df['SLG'] - df['BA']
# batting average on balls in play (related to luck)
BABIP_numer = df['H'] - df['HR']
BABIP_denom = df['AB'] - df['SO'] - df['HR'] + df['SF']
df['BABIP'] = (BABIP_numer / BABIP_denom).round(decimals=3)
# record potentially non-linear age curve
df['Age_Squared'] = df['Age'] ** 2

# cols that we will accumulate means of
# est_ba == xBA
cum_cols = ['BA', 'est_ba', 'bbe']

# add team's average ba for each season
team_cols = [c for c in df.columns if c.startswith('TEAM_')]
team_names = df[team_cols].idxmax(axis=1)
df['Temp_TeamName'] = team_names.str.replace('TEAM_', '')
# calculate the mean BA for each TeamName within each year
team_avg_ba = df.groupby(['year', 'Temp_TeamName'])['BA'].transform('mean')
df['TeamAvg_BA'] = team_avg_ba 
df = df.drop('Temp_TeamName', axis=1)

# add cumulative stats for each players' career
# expanding starts at beginning of each group and grows
# calculate expanding mean
cumulative_stats_raw = df.groupby('key_bbref')[cum_cols].expanding().mean()
# reset the index to put the original index back and align with the main DF
cumulative_stats_aligned = cumulative_stats_raw.reset_index(level=[0, 1], drop=False)
# gives year N's row the cumulative average up to year N-1.
cumulative_stats_shifted = cumulative_stats_aligned.groupby('key_bbref')[cum_cols].shift(1)
# concat the shifted results back to the main DF
cumulative_stats_shifted.columns = [f'CareerAvg_{c}' for c in cum_cols]
df = pd.concat([df, cumulative_stats_shifted], axis=1)

# add previous years' averages (not cumulative)
stats_to_lag = [col for col in df.columns if col != 'key_bbref'
    and 'POS_' not in col
    and 'TEAM_' not in col
    and 'CareerAvg_' not in col]
grouped = df.groupby('key_bbref')[stats_to_lag]
PREV_YEARS = 3
# record previous PREV_YEARS years' stats
for i in range(1,PREV_YEARS+1):
    shifted = grouped.shift(i)
    shifted.columns = [f'{col}_lag{i}' for col in stats_to_lag]
    df = pd.concat([df, shifted], axis=1)
#  fill missing of previous n years
for col in stats_to_lag:
    for i in range(2,PREV_YEARS+1):
        df[f'{col}_lag{i}'] = df[f'{col}_lag{i}'].fillna(df[f'{col}_lag{i-1}'])

# select features
features_to_keep = [c for c in df.columns if 'lag' in c or 'CareerAvg' in c]
# keeping age isnt cheating
features_to_keep += ['Age', 'Age_Squared']
# drop rookies
df = df.dropna()

print(f'final row len {len(df[features_to_keep])}')
print(f'final col len {len(df[features_to_keep].columns)}\n')

TRAIN_MIN = 2015
TRAIN_MAX = 2023
TEST_MAX = 2024
if TRAIN_MAX == 2023 and TEST_MAX == 2024:
    # special case 1: record which players we are testing on
    # for 2024 in order to compare with batx predictions
    df_temp = df[(df['year'] > TRAIN_MAX) & (df['year'] <= TEST_MAX)]
    df_temp = df_temp['key_bbref']
    # inner join this with './ID_Mapping/mapping_fan.csv'
    FAN_PATH = './ID_Mapping/mapping_fan.csv'
    df_fan_map = pd.read_csv(FAN_PATH)
    df_fan_map = pd.merge(df_fan_map, df_temp)
    df_fan_map.to_csv(FAN_PATH, index=False)
    print(f'len of df_fan_map {len(df_fan_map)}')
elif TRAIN_MAX == 2024 and TEST_MAX == 2025:
    # special case 2: record which players we are testing on
    # for 2025 in order to compare with steamer predictions
    df_temp = df[(df['year'] > TRAIN_MAX) & (df['year'] <= TEST_MAX)]
    df_temp = df_temp['key_bbref']
    # inner join this with './ID_Mapping/mapping_fan.csv'
    FAN_PATH = './ID_Mapping/steamer_mapping.csv'
    df_map = pd.read_csv(FAN_PATH)
    df_map = pd.merge(df_map, df_temp)
    df_map.to_csv(FAN_PATH, index=False)
    print(f'len of df_map{len(df_map)}')

# set this year's stats as previous year's stats
X = df[features_to_keep]
y = df['BA']

# split
X_train = X[(TRAIN_MIN <= df['year']) & (df['year'] <= TRAIN_MAX)]
y_train = y[(TRAIN_MIN <= df['year']) & (df['year'] <= TRAIN_MAX)]
X_test = X[(df['year'] > TRAIN_MAX) & (df['year'] <= TEST_MAX)]
y_test = y[(df['year'] > TRAIN_MAX) & (df['year'] <= TEST_MAX)]
print(f'train len {len(X_train)}')
print(f'test len {len(X_test)}')

# scale for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# PCA
pca = PCA(n_components=0.89)  # keep some % of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
''' # visualize top k selected
selector = SelectKBest(score_func=f_regression, k=25)
selector.fit(X_train_scaled, y_train)
scores = pd.Series(selector.scores_, index=X_train.columns)
top_scores = scores.sort_values(ascending=False).head(25)
table_df = top_scores.to_frame(name='F-Score').reset_index()
table_df.columns = ['Feature Name', 'F-Score'] 
# round the f-Score for cleaner visualization
table_df['F-Score'] = table_df['F-Score'].round(2)
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')
ax.axis('tight')
table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()
'''
# select top k for training and testing
# X_train = X_train_pca = selector.transform(X_train_scaled)
# X_test = X_test_pca = selector.transform(X_test_scaled)

# linreg train
linear_model = LinearRegression()
linear_model.fit(X_train_pca, y_train)
y_pred = linear_model.predict(X_test_pca)
# linreg MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Linear Regression MAE: {mae}')

# XGB init
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    # objective='reg:absoluteerror', # 'not guaranteed to be optimal' 
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
)
# XBG train
xgb_model.fit(X_train_pca, y_train)
y_pred = xgb_model.predict(X_test_pca)
# XGB MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'XGBoost MAE: {mae}')

# NN init
mlp_model = MLPRegressor(
    hidden_layer_sizes=([50] * 20),
    activation='relu',
    solver='adam'
)
# NN train
mlp_model.fit(X_train_pca, y_train)
y_pred = mlp_model.predict(X_test_pca)
# NN MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'NN MAE: {mae}')

# lasso init (cv aluto finds optimal alpha)
lasso_model = LassoCV(max_iter=1000000)
# lasso train
lasso_model.fit(X_train_pca, y_train)
# lasso MAE
y_pred = lasso_model.predict(X_test_pca)
mae = mean_absolute_error(y_test, y_pred)
print(f'lasso MAE: {mae}')
