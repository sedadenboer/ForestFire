from plot import ignition_vs_ratio_heatmap
import json
import numpy as np

step = 0.05
varying_ignition = np.arange(0, 1 + step, step)

igni_ratios = []
for i in range(5):
    with open(rf'C:\Users\wangk\Desktop\css_resources\Codes\ForestFire\Code\Output\Heatmaps\perco_density_0.75\perco_rep{i+1}.json', 'r') as fp:
        igni_ratio = json.load(fp)
        igni_ratios.append(igni_ratio)
        
# compute average
avg_perco_rep = {}
for key in igni_ratios[0].keys():
    igni_ratio = np.zeros(len(igni_ratios[0].keys()))
    for i in range(5):
        igni_ratio += np.array(igni_ratios[i][key])

    avg_perco_rep[key] = list(igni_ratio / 5)

filename = 'perco_prob'
ignition_vs_ratio_heatmap(avg_perco_rep, varying_ignition, filename, savefig=False)


################### Area plot ###################
area_ratios = []
for i in range(5):
    with open(rf'C:\Users\wangk\Desktop\css_resources\Codes\ForestFire\Code\Output\Heatmaps\area_density_0.75\area_rep{i+1}.json', 'r') as fp:
        area_ratio = json.load(fp)
        area_ratios.append(area_ratio)
        
# compute average
avg_area_rep = {}
for key in area_ratios[0].keys():
    area_ratio = np.zeros(len(area_ratios[0].keys()))
    for i in range(5):
        area_ratio += np.array(area_ratios[i][key])

    avg_area_rep[key] = list(area_ratio / 5)
    
filename = 'area'
ignition_vs_ratio_heatmap(avg_area_rep, varying_ignition, filename, savefig=False)
    
