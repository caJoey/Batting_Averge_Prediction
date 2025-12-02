from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

df = pd.read_csv('Get_Stats/stats.csv')
df = df.sort_values(by=['key_bbref', 'year']).reset_index(drop=True)

cols = [
    'Age', 'WAR', 'G', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'RBI', 'SO',
    'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'rOBA', 'Rbat+', 'HBP', 'SH', 'SF', 'IBB'
]
# add previous year's result to every row
for col in cols:
    df[f'{col}_lag1'] = df.groupby('key_bbref')[col].shift(1)
# record previous year's stats
lagged_features = [f'{col}_lag1' for col in cols]
# remove players who only played 1 season
df = df.dropna()
print(len(df[df['year'] == 2025]))
cutoff_year = 2025
# set this year's stats as previous year's stats
X = df[lagged_features]
y = df['BA']
# split
X_train = X[df['year'] < cutoff_year]
y_train = y[df['year'] < cutoff_year]
X_test = X[df['year'] >= cutoff_year]
y_test = y[df['year'] >= cutoff_year]

# train
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

# error
mae = mean_absolute_error(y_test, y_pred)
print(f'mae {mae}')
