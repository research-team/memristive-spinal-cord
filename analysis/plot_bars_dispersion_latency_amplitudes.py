from analysis.functions import read_NEURON_data, dict_to_list, find_mins
import numpy as np
from matplotlib import pylab as plt
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.peaks_of_real_data_without_EES import delays
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
neuron_data_21cms_dict = read_NEURON_data\
    ('../../neuron-data/15cm.hdf5')
neuron_data_21cms_list = dict_to_list(neuron_data_21cms_dict)
neuron_data_21cms_means = list(map(lambda x: np.mean(x), zip(*neuron_data_21cms_list)))
slices_start_time_from_EES_neuron = find_mins(neuron_data_21cms_means, -13)[1]
step_neuron = slices_start_time_from_EES_neuron[1] - slices_start_time_from_EES_neuron[0]
slices_start_time_neuron = []
slices_start_time_neuron.append(0)
for i in range(len(slices_start_time_from_EES_neuron)):
    slices_start_time_neuron.append(slices_start_time_neuron[i] + step_neuron)
print("len(neuron_data_21cms_list) = ", len(neuron_data_21cms_list[0]))
# plt.plot(neuron_data_21cms_list[10])
# for i in slices_start_time_neuron:
#     plt.axvline(x=i, linestyle='--', color='gray')
#     plt.axhline(y=-9, linestyle='--', color='gray')
#     plt.axhline(y=-2, linestyle='--', color='gray')
# plt.show()
data_step = 0.025
data_neuron_s21cms = calc_max_min(slices_start_time_neuron, neuron_data_21cms_means, data_step)
print("data_neuron_s21cms = ", len(data_neuron_s21cms[0]), data_neuron_s21cms[2])
# for index in data_neuron_s21cms[2].values():
#     for i in range(len(index)):
#         if index[i] < 10:
#             del data_neuron_s21cms[2][i + 1]
#             del data_neuron_s21cms[3][i + 1]
# print("data_neuron_s21cms[2] = ", data_neuron_s21cms[2])
latency_neuron_21 = delays(data_neuron_s21cms[0], data_neuron_s21cms[2], data_neuron_s21cms[3], -8, -3.8, 'neuron')[1]
# print("latency_neuron_21 = ", latency_neuron_21)
neuron_21_list_all_runs = []
for run in range(len(neuron_data_21cms_list)):
    neuron_21_list_all_runs.append(calc_max_min(slices_start_time_neuron, neuron_data_21cms_list[run], data_step))
neuron_21_latencies_all_runs = []
for run in range(len(neuron_21_list_all_runs)):
    print("run = ", run)
    neuron_21_latencies_all_runs.append(delays(neuron_21_list_all_runs[run][0], neuron_21_list_all_runs[run][2],
                                               neuron_21_list_all_runs[run][3], -9, -2, 'neuron'))
for run in range(len(neuron_21_latencies_all_runs)):
    if(len(neuron_21_latencies_all_runs[run][1]) < 12):
        neuron_21_latencies_all_runs[run][1]
print("neuron_21_latencies_all_runs = ", len(neuron_21_latencies_all_runs[10][1]), neuron_21_latencies_all_runs[10][1])
# print("len(neuron_21_latencies_all_runs) = ", len(neuron_21_latencies_all_runs[0][1]))  # 100   # 4 # 12
latencies_mins_neuron_21_all_runs = []
all_latencies_mins_neuron_21_all_runs = []
# print("neuron_21_list_all_runs = ", len(neuron_21_list_all_runs[0][0]))
# plt.plot(neuron_data_21cms_list[10])
# for sl in slices_start_time_neuron:
#     plt.axvline(x=sl, linestyle='--', color='gray')
# plt.axhline(y=-9, linestyle='--', color='gray')
# plt.axhline(y=-2, linestyle='--', color='gray')
# plt.show()
# print(len(neuron_21_latencies_all_runs[17][1]), "neuron_21_latencies_all_runs[17] = ",
#       neuron_21_latencies_all_runs[17][1])
r = 0
for sl in range(len(neuron_21_latencies_all_runs[r][1])):
    latencies_mins_neuron_21_all_runs_tmp = []
    all_latencies_mins_neuron_21_all_runs_tmp = []
    for run in range(len(neuron_21_latencies_all_runs)):
        # print("run = ", run)
        # print("sl = ", sl)
        # print("len(neuron_21_latencies_all_runs[run][1]) = ", len(neuron_21_latencies_all_runs[run][1]))
        latencies_mins_neuron_21_all_runs_tmp.append(neuron_21_latencies_all_runs[run][1][sl])
        all_latencies_mins_neuron_21_all_runs_tmp.append(neuron_21_latencies_all_runs[run][3][sl])
    r += 1
    latencies_mins_neuron_21_all_runs.append(latencies_mins_neuron_21_all_runs_tmp)
    all_latencies_mins_neuron_21_all_runs.append(all_latencies_mins_neuron_21_all_runs_tmp)
# print("latencies_mins_neuron_21_all_runs = ", len(latencies_mins_neuron_21_all_runs[0]))
# print(len(latencies_mins_neuron_21_all_runs[0]), "latencies_mins_neuron_21_all_runs = ",
#       latencies_mins_neuron_21_all_runs)
all_latencies_mins_neuron_21_all_runs.append(all_latencies_mins_neuron_21_all_runs_tmp)
amplitudes = []
# print("len(all_latencies_mins_neuron_21_all_runs) = ", len(all_latencies_mins_neuron_21_all_runs[0]))
for sl in range(len(all_latencies_mins_neuron_21_all_runs)):
    amplitudes_tmp = []
    for run in range(len(all_latencies_mins_neuron_21_all_runs[sl])):
        # print("sl = ", sl)
        # print("run = ", run)
        amplitudes_tmp.append(abs(np.mean(all_latencies_mins_neuron_21_all_runs[sl][run])))
    amplitudes.append(amplitudes_tmp)
print("amplitudes = ", amplitudes)
for i in range(len(latencies_mins_neuron_21_all_runs)):
    print("amplitudes[{}] = ".format(i), amplitudes[i])
list_to_draw = []
list_to_draw.append(latencies_mins_neuron_21_all_runs)
list_to_draw.append(amplitudes)
xs = []
for i in range(len(latencies_mins_neuron_21_all_runs)):
    for j in range(len(latencies_mins_neuron_21_all_runs[i])):
        xs.append(i + 1)
for i in range(len(xs)):
    xs.append(xs[i])
col_latencies = []
col_amplitudes = []
for i in range(len(latencies_mins_neuron_21_all_runs)):
    for j in range(len(latencies_mins_neuron_21_all_runs[i])):
        col_latencies.append(latencies_mins_neuron_21_all_runs[i][j])
        col_amplitudes.append(amplitudes[i][j])
print("max(col_latencies) = ", max(col_latencies))
print("min(col_latencies) = ", min(col_latencies))
name = []
for i in range(len(col_latencies)):
    name.append('Latencies')
print("col_latencies = ", col_latencies)
for i in range(len(col_amplitudes)):
    col_latencies.append(col_amplitudes[i])
print("col_latencies = ", col_latencies)
print("max(col_latencies) = ", max(col_latencies))
print("min(col_latencies) = ", min(col_latencies))
for i in range(len(col_amplitudes)):
    name.append('Amplitudes')
print(name)
df = pd.DataFrame({'Slice':xs})
df['Latencies, mV'] = col_latencies
df['name'] = name
# print("df = ", df)
# for i in range(len(latencies_mins_neuron_21_all_runs)):
    # ax = sns.boxplot(x=latencies_mins_neuron_21_all_runs, data=df, color='#F2AA2E')
ax = sns.boxplot(x='Slice', y='Latencies, mV', data=df, hue='name')
# b = sns.boxplot(x=df["Latencies, mV"])
# b.set_xlabel("Slice", fontsize=28)
# b.set_ylabel("Latencies, mV", fontsize=28)
sns.set_context(font_scale=28)
plt.tick_params(labelsize=28)
plt.legend(fontsize=28)
plt.rc('font', size = 28)
plt.rc('legend', fontsize=28)
plt.show()