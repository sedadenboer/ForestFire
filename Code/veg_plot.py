import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import sys

# Load data
ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
burn_ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = []

# C:\Users\wangk\Desktop\css_resources\Codes\ForestFire\Code\Output\size100_1\cript_p_b1_r_0.0_1.0.json

for j in burn_ts:
    for i in ratio:
        with open(rf'C:/Users/wangk/Desktop/css_resources/Codes/ForestFire/Code/Output/size100_1/cript_p_b{j}_r_{i}_{round(1.0-i, 1)}.json', 'r') as fp:
            d = json.load(fp)
            data.append(d)

print(len(data))

# compute the average percolation density
data_avg_ps = data.copy()
for i, d in enumerate(data):
    keys = d.keys()
    for key in keys:
        avg_prob = np.mean(np.array(d[key]))
        data_avg_ps[i][key] = avg_prob

print(data_avg_ps[0])

# create the critical density dataset
data_crit_p = []
keys = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
values = np.zeros(11)
for i in range(10):
    data_crit_p.append(dict(zip(keys, values)))

for count, d in enumerate(data_avg_ps):
    for item in d.items():
        if item[1] == 0.0:
            data_crit_p[count // 11][keys[count % 11]] = np.nan
        if item[1] >= 0.5:
            data_crit_p[count // 11][keys[count % 11]] = item[1]
            break

print(data_crit_p[0])

# sys.exit()

# for plotting

fig, ax = plt.subplots(1)

for i, d in enumerate(data_crit_p):
    # create a DataFrame from the dictionary
    df = pd.DataFrame(d, index=[1])
    print(df)
    if not df.isnull().values.all():
        # reset the index and melt the DataFrame
        df = df.reset_index().melt(id_vars='index', var_name='tree ratio', value_name='Probability')
        # rename the columns
        df = df.rename(columns={'index': 'Experiment Number'})

        sns.set_style("ticks")
        sns.lineplot(data=df, x='tree ratio', y='Probability',
                    markers=True, dashes=False, ax=ax, label=i)
        plt.xlabel('tree ratio')
        plt.ylabel('critical density')


plt.show()
