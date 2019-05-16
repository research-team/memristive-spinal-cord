import pandas as pd
from matplotlib import pylab as plt


def plot_hyst(path):
	df = pd.read_csv(path)
	ys = df['Reading']
	xs = df['Value']
	print("ys = ", ys)
	print("xs = ", xs)
	return xs, ys


up = plot_hyst('/home/anna/Downloads/m2019031403.csv')
# up2 = plot_hyst('/home/anna/Downloads/memmid0405up1.csv')
# down = plot_hyst('/home/anna/Downloads/memmid0404dow1.csv')
plt.plot(up[0], up[1], color ='green', label='up')
# plt.plot(up2[0], up2[1], color ='grey', label='up')
# plt.plot(down[0], down[1], color='blue', label='down')
# plt.legend()
plt.show()