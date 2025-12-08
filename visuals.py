'''
Visualize stuff for the paper / slideshow
ohtansh01
'''

import pandas as pd
import matplotlib.pyplot as plt

STATS_PATH = './Get_Stats/stats.csv'
df_stats = pd.read_csv(STATS_PATH)
# Shohei Ohtani's Standard Stats From 2025
'''
shohei_stats_2025 = df_stats[(df_stats['year'] == 2025) & (df_stats['key_bbref'] == 'ohtansh01')]
print(df_stats.head())
row_series = shohei_stats_2025.iloc[0]
table_df = row_series.to_frame().reset_index()
table_df.columns = ['Stat Name', 'Stat']
fig, ax = plt.subplots(figsize=(6, 10)) 
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

# 660271
ADV_STATS_PATH = './Get_Adv_Stats/adv_stats.csv'
df_adv_stats = pd.read_csv(STATS_PATH)
# 1. Shohei Ohtani's Advanced Stats From 2025
shohei_stats_2025 = df_stats[(df_stats['year'] == 2025) & (df_stats['key_bbref'] == 'ohtansh01')]
print(df_stats.head())
row_series = shohei_stats_2025.iloc[0]
table_df = row_series.to_frame().reset_index()
table_df.columns = ['Stat Name', 'Stat']
fig, ax = plt.subplots(figsize=(6, 10)) 
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

# Test set results 2023-2025
'''
data = {
    'LR': 0.020809153140335167,
    'XGB': 0.022848563880311472,
    'NN': 0.02551761481368845,
    'Lasso': 0.021122842368615388
}
models = list(data.keys())
scores = list(data.values())
plt.figure(figsize=(7, 5))
bars = plt.bar(models, scores, color=['blue', 'orange', 'green', 'purple'])
# Add MAE labels on top of the bars
for i, score in enumerate(scores):
    # 5 decimal places
    plt.text(i, score + 0.0001, f'{score:.5f}', ha='center', va='bottom', fontsize=9)
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('2023 - 2025 Test MAE')
plt.ylim(min(scores) * 0.95, max(scores) * 1.1)
plt.show()
'''

# vs. Steamer for 2025 test set
'''
data = {
    'LR': 0.0201370995339794,
    'XGB': 0.021906903164916572,
    'NN': 0.020972327185268317,
    'Lasso': 0.020413419606862958,
    'Steamer': 0.0219409
}
models = list(data.keys())
scores = list(data.values())
plt.figure(figsize=(7, 5))
bars = plt.bar(models, scores, color=['blue', 'orange', 'green', 'purple', 'red'])
# Add MAE labels on top of the bars
for i, score in enumerate(scores):
    # 5 decimal places
    plt.text(i, score + 0.0001, f'{score:.5f}', ha='center', va='bottom', fontsize=9)
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Steamer 2025 Comparison')
plt.ylim(min(scores) * 0.95, max(scores) * 1.1)
plt.show()
'''

'''
# vs. The Bat X for 2024 test set
data = {
    'LR': 0.02143809110381047,
    'XGB': 0.023090864918771246,
    'NN': 0.024612124644566174,
    'Lasso': 0.02162984251670102,
    'The Bat X': 0.022349136904761906
}
models = list(data.keys())
scores = list(data.values())
plt.figure(figsize=(7, 5))
bars = plt.bar(models, scores, color=['blue', 'orange', 'green', 'purple', 'red'])
# Add MAE labels on top of the bars
for i, score in enumerate(scores):
    # 5 decimal places
    plt.text(i, score + 0.0001, f'{score:.5f}', ha='center', va='bottom', fontsize=9)
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('The Bat X 2024 Comparison')
plt.ylim(min(scores) * 0.95, max(scores) * 1.1)
plt.show()
'''
