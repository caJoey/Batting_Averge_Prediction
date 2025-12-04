from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv('Get_Stats/stats.csv')
df = df.sort_values(by=['key_bbref', 'year']).reset_index(drop=True)
# cols = [
#     'Age', 'WAR', 'G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'SO',
#     'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'rOBA', 'Rbat+', 'HBP', 'SH', 'SF', 'IBB', 'BABIP'
# ]

cols = [
    'BA', 'OBP', 'SLG', 'K%', 'BB%', 'ISO', 'BABIP', 'Age', 'Age_Squared',
    'OPS+', 'IBB', 'RBI', 'rOBA', 'Rbat+', 'WAR', 'PA'
]

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
# add season's averages
season_means = df.groupby('year')[cols].transform('mean')
season_means.columns = [f'LeagueAvg_{c}' for c in season_means.columns]
df = pd.concat([df, season_means], axis=1)
# add cumulative average for each players' career
# expanding starts at beginning of each group and grows
cumulative_stats = df.groupby('key_bbref')[cols].expanding().mean().reset_index(level=0, drop=True)
cumulative_stats = df.groupby('key_bbref')[cols].shift(1) # Shift to prevent data leakage
cumulative_stats.columns = [f'CareerAvg_{c}' for c in cols]
df = pd.concat([df, cumulative_stats], axis=1)
# add previous years' averages
# stats_to_lag = cols + [f'LeagueAvg_{c}' for c in cols if f'LeagueAvg_{c}' in df.columns]
stats_to_lag = cols
grouped = df.groupby('key_bbref')[stats_to_lag]
PREV_YEARS = 8
# record previous 3 years' stats
for i in range(1,PREV_YEARS+1):
    shifted = grouped.shift(i)
    shifted.columns = [f'{col}_lag{i}' for col in stats_to_lag]
    df = pd.concat([df, shifted], axis=1)
#  fill missing of previous n years
for col in stats_to_lag:
    for i in range(2,PREV_YEARS+1):
        df[f'{col}_lag{i}'] = df[f'{col}_lag{i}'].fillna(df[f'{col}_lag{i-1}'])
# Drop rookies
df = df.dropna()
# Select features
features_to_keep = [c for c in df.columns if 'lag' in c or 'CareerAvg' in c]
# keeping age isnt cheating
features_to_keep += ['Age', 'Age_Squared']
print(f'Total Rows Available: {len(df)}')

# # add previous year's result to every row
# for col in cols:
#     df[f'{col}_lag'] = df.groupby('key_bbref')[col].shift(1)
# # record previous year's stats
# lagged_features = [f'{col}_lag' for col in cols]
# # remove players who only played 1 season
# df = df.dropna()



TRAIN_MIN = 1955
TRAIN_MAX = 2015
# set this year's stats as previous year's stats
X = df[features_to_keep]
for col in X.columns:
    pass
    print(col)
y = df['BA']
# split
X_train = X[(TRAIN_MIN <= df['year']) & (df['year'] < TRAIN_MAX)]
y_train = y[(TRAIN_MIN <= df['year']) & (df['year'] < TRAIN_MAX)]
X_test = X[df['year'] >= TRAIN_MAX]
y_test = y[df['year'] >= TRAIN_MAX]

# linreg train
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
# linreg MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Linear Regression MAE: {mae}')

# XGB init
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    # objective='reg:absoluteerror', # 'not guaranteed to be optimal' 
    n_estimators=500,
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp_model = MLPRegressor(
    hidden_layer_sizes=([50] * 20),
    activation='relu',
    solver='adam'
)
# NN train
mlp_model.fit(X_train_scaled, y_train)
y_pred = mlp_model.predict(X_test_scaled)
# NN MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'NN MAE: {mae}')

# lasso init (cv aluto finds optimal alpha)
lasso_model = LassoCV()
# lasso train
lasso_model.fit(X_train, y_train)
# lasso MAE
y_pred = lasso_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'lasso MAE: {mae}')
