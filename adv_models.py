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
# inner join on key_mlbam
map_adv = pd.merge(df_map, df_adv_stats)
# inner join on [key_bbref, key_mlbam]
df = pd.merge(stats_map, map_adv)
# drop stuff we don't care about for machine learning
df = df.drop(['player', 'key_mlbam'], axis=1)
# for col in df.columns:
#     print(col)
df = df.dropna()
print(df.head())
print(f'row len {len(df)}')
print(f'col len {len(df.columns)}')
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

cols = [col for col in df.columns if col != 'key_bbref'
    and 'POS_' not in col and 'TEAM_' not in col]

# add season's averages for each season
season_means = df.groupby('year')[cols].transform('mean')
season_means.columns = [f'LeagueAvg_{c}' for c in season_means.columns]
df = pd.concat([df, season_means], axis=1)

# add team's average ba for each season
team_cols = [c for c in df.columns if c.startswith('TEAM_')]
team_names = df[team_cols].idxmax(axis=1)
df['Temp_TeamName'] = team_names.str.replace('TEAM_', '')
# calculate the mean BA for each TeamName within each year
team_avg_ba = df.groupby(['year', 'Temp_TeamName'])['BA'].transform('mean')
df['TeamAvg_BA'] = team_avg_ba 
df = df.drop('Temp_TeamName', axis=1)
cols.append('TeamAvg_BA')

# add cumulative batting average for each players' career
# expanding starts at beginning of each group and grows
cumulative_stats = df.groupby('key_bbref')[cols].expanding().mean().reset_index(level=0, drop=True)
cumulative_stats = df.groupby('key_bbref')[cumulative_stats.columns].shift(1)
cumulative_stats.columns = [f'CareerAvg_{c}' for c in cols]
df = pd.concat([df, cumulative_stats], axis=1)

# add previous years' averages
# stats_to_lag = cols + [f'LeagueAvg_{c}' for c in cols if f'LeagueAvg_{c}' in df.columns]
stats_to_lag = [col for col in df.columns if col != 'key_bbref'
    and 'POS_' not in col
    and 'TEAM_' not in col
    and 'CareerAvg' not in col]
grouped = df.groupby('key_bbref')[stats_to_lag]
PREV_YEARS = 8
# record previous PREV_YEARS years' stats
for i in range(1,PREV_YEARS+1):
    shifted = grouped.shift(i)
    shifted.columns = [f'{col}_lag{i}' for col in stats_to_lag]
    df = pd.concat([df, shifted], axis=1)

#  fill missing of previous n years
for col in stats_to_lag:
    for i in range(2,PREV_YEARS+1):
        df[f'{col}_lag{i}'] = df[f'{col}_lag{i}'].fillna(df[f'{col}_lag{i-1}'])

# Select features
features_to_keep = [c for c in df.columns if 'lag' in c or 'CareerAvg' in c]
# keeping age isnt cheating
features_to_keep += ['Age', 'Age_Squared']
# drop rookies
df = df.dropna()
TRAIN_MIN = 2015
TRAIN_MAX = 2024
TEST_MAX = 2025
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
    print(f'len {len(df_fan_map)}')
elif TRAIN_MAX == 2024 and TEST_MAX == 2025:
    # special case 2: record which players we are testing on
    # for 2025 in order to compare with steamer predictions
    df_temp = df[(df['year'] > TRAIN_MAX) & (df['year'] <= TEST_MAX)]
    df_temp = df_temp['key_bbref']
    # inner join this with './ID_Mapping/mapping_fan.csv'
    FAN_PATH = './ID_Mapping/steamer_mapping.csv'
    df_fan_map = pd.read_csv(FAN_PATH)
    df_fan_map = pd.merge(df_fan_map, df_temp)
    df_fan_map.to_csv(FAN_PATH, index=False)
    print(f'len {len(df_fan_map)}')

# set this year's stats as previous year's stats
X = df[features_to_keep]
for col in X.columns:
    pass
    # print(col)
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
pca = PCA(n_components=0.97)  # keep some % of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# selector
# selector = SelectKBest(score_func=f_regression, k=50)
# selector.fit(X_train_scaled, y_train)
# scores = pd.Series(selector.scores_, index=X_train.columns)
# print(scores.sort_values(ascending=False).head(50))
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
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
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
