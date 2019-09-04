from analysis.functions import read_data, calc_boxplots
import pandas as pd
import seaborn as sns
import pylab as plt
import numpy as np

first = read_data('/home/anna/Downloads/Telegram Desktop/1.hdf5')
second = read_data('/home/anna/Downloads/Telegram Desktop/2.hdf5')
third = read_data('/home/anna/Downloads/Telegram Desktop/3.hdf5')

all_runs = first + second + third
# plt.plot(all_runs[0])
# plt.show()
print(len(all_runs), len(all_runs[0]))

final = []
# plt.figure()
for j in range(11000):   # 6000
	bp_size = []
	border = 2
	print("j = ", j)
	bp = []
	list_for_bp = []
	offset = 0
	length = []
	for k in range(len(all_runs) - 1):
		# print("k = ", k)
		for i in range(offset, border):
			# print("offset = ", offset)
			# print("border = ", border)
			# print("---")
			list_for_bp.append(all_runs[i][j])
			offset += 2
		length.append(len(list_for_bp))
		# print("list_for_bp = ", len(list_for_bp), list_for_bp)
		bp.append(calc_boxplots(list_for_bp))
		# print("length = ", length)
		# if len(length) > 1:
		# 	if length[-1] > length[-2]:
		# 		bp.append(calc_boxplots(list_for_bp))
		# 		print("length inside = ", length)
		# print("bp = ", len(bp), bp)
		border += 1

	for b in range(len(bp)):
		bp_size.append(bp[b][1] - bp[b][2])
	final.append(bp_size)
print("final = ", len(final))
final = np.array(final)
print(len(final))
final = final.T
print(len(final))
	# raise Exception
# print("len(bp_size) = ", len(bp_size), bp_size[:100])
# raise Exception
# bp_size = [bp_size[x:x + len(all_runs)] for x in range(0, len(bp_size), len(all_runs))]
# bp_size = np.array(bp_size)
# bp_size = bp_size.T

final = final.flatten()
print("len(final) = ", len(final))
# raise Exception
labels = []
for i in range(int(len(final) / 11000)):
	for j in range(11000):
		labels.append('{} / {}'.format(i + 2, i + 3))

df = pd.DataFrame({'Values': final, 'Labels': labels})
# i = 0
# for f in final:
# 	print("i = ", i)
# 	sns.boxplot(data=f)
# 	i += 1

# print('hmmm')
# plt.show()
# labels = []
# for i in range(int(len(bp_size) / len(all_runs[0]))):
# 	for j in range(len(all_runs[0])):
# 		labels.append('{} / {}'.format(i + 2, i + 3))
# df = pd.DataFrame({'Values': bp_size, 'Labels': labels})
sns.boxplot(x='Labels', y='Values', data=df)
plt.show()