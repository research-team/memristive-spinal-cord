from stat_analysis.averaging_data import averaging_data
import matplotlib.pyplot as plt

path = ''
pattern = '*E*13.5*'

data_mean = averaging_data(path, pattern)
plt.plot(data_mean, linewidth=5)
plt.show()
# print(np.concatenate(data_mean).shape)
# print(data_mean_foot.shape)
#
# print(stats.ks_2samp(np.concatenate(data_mean_qpz), np.concatenate(data_mean_foot)))
# #ответ: Ks_2sampResult(statistic=1.0, pvalue=0.0)




