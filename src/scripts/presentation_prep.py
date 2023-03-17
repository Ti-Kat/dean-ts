import pandas as pd
import numpy as np

df = pd.read_csv(
    "../datasets/nab/art_daily_flatmiddle.csv",
    index_col='timestamp',
    parse_dates=['timestamp'],
)
df['time'] = np.arange(len(df.index))
df.head()

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
# %config InlineBackend.figure_format = 'retina'

ab = np.random.choice(range(-20, 20), size=80)
ac = ab.copy()
ac = ac + 3

ab[0:5] = -ab[0:5]
ab[30:40] = -ab[30:40]
print(ab)
print(ac)

pts_array = [0, 1, 2, 3, 4]
opp_pts_array = [3, 4, 5, 6, 7]

# d = {'points':ab,'opponent_points':ac}
# dfscatter = pd.DataFrame(d)
#
# dfscatter.plot.scatter(x='points', y='opponent_points')

# ts = pd.Series(ab)
# df = pd.DataFrame(ac, index=ts.index)
#
# plt.figure(figsize=(12,5))
#
# ax1 = df.A.plot(color='blue', grid=True, label='Count')
# ax2 = df.B.plot(color='red', grid=True, secondary_y=True, label='Sum')
#
# h1, l1 = ax1.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
#
#
# plt.legend(h1+h2, l1+l2, loc=2)
# plt.show()

# fig, ax = plt.subplots()
# ax.plot('time', 'value', data=df_comb, color='0.75')
# ax = sns.scatterplot(x='time', y='value', data=df_comb)
# # ax.set_xticks(range(1, df_head.shape[0]))
# # ax.set_title('Plot of art daily flat middle')
# plt.show()
#
df_head = df.iloc[100:125]
df_tail = df.iloc[200:242]
# print(df_head.shape)
# exit()
df_head['time'] -= 100
df_tail['time'] -= 175
df_comb = df_head.append(df_tail)
df_comb.loc[df_comb['time'] == 60, 'value'] = 80
df_comb.loc[df_comb['time'] > 63, 'value'] += 100
print(df_head)
#
fig, ax = plt.subplots()
ax.plot('time', 'value', data=df_comb, color='0.75')
ax = sns.scatterplot(x='time', y='value', data=df_comb)
# ax.set_xticks(range(1, df_head.shape[0]))
# ax.set_title('Plot of art daily flat middle')
plt.show()
