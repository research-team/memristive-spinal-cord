from analysis.functions import read_data
from matplotlib import pylab as plt
import numpy as np

gras_foot = '/home/anna/Desktop/data/gras/foot/gras_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5'
gras_4pedal =

bio_str = '/home/anna/Desktop/data/bio/str/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5'
bio_foot = '/home/anna/Desktop/data/bio/foot/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5'
bio_4pedal = '/home/anna/Desktop/data/bio/4pedal/bio_E_13.5cms_40Hz_i100_4pedal_no5ht_T_0.25step.hdf5'
bio_toe = '/home/anna/Desktop/data/bio/toe/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5'
bio_air = '/home/anna/Desktop/data/bio/air/bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5'
bio_qpz= '/home/anna/Desktop/data/bio/qpz/bio_E_13.5cms_40Hz_i100_2pedal_5ht_T_0.1step.hdf5'

# don't forget to normalize data

gras = read_data(gras_foot)
# gras = gras[:11000]
print(len(gras), len(gras[0]))

# for i in range(len(gras)):
# 	plt.plot(gras[i])
# 	plt.show()
mean_1 = np.mean(gras[0])
mean_2 = np.mean(gras[1])

stds = []
for i in range(len(gras)):
	stds.append(np.std(gras[i][5000:11000]))

print(len(stds))
std_1 = np.std(gras[0])
# std_2 = np.std(gras[2])

for s in stds:
	print("s = ", s)
# print("mean_1 = ", mean_1)
# print("mean_2 = ", mean_2)

# print("std_1 = ", std_1)
# print("std_2 = ", std_2)
# plt.plot(gras[0])
# plt.show()

array = 0
for a in stds:
	array += ((1.96 * 1.96) *((a* a) / (0.95 * 0.95)))

final = array / len(gras)
print(final)

n = 0
z = 1.96 * 1.96
e = 0.1
for a in stds:
	n += ((z * z) * (a * a)) / (e * e)

final = n / len(gras)
# print(final)    # biostr = 8332.593938689417

#gras = 2697.8859993373067