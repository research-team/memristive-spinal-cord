import numpy as np
import pylab as plt
from scipy.io import loadmat


def plot_fig(filename, title, rat, begin, end):
	d = loadmat(filename, squeeze_me=True, struct_as_record=False)
	ax1 = d['hgS_070000'].children

	if np.size(ax1) > 1:
		ax1 = ax1[0]

	plt.figure(figsize=(16, 9))

	yticks = []
	plt.suptitle(f"{title} [{begin} : {end}] \n {rat}")

	for i, line in enumerate(ax1.children, 1):
		if line.type == 'graph2d.lineseries':
			if begin <= i <= end:
				color = 'r'
			else:
				color = 'gray'

			x = line.properties.XData
			y = line.properties.YData
			yticks.append(y[0])

			plt.plot(x, y, color=color)
		if line.type == 'text':
			break

	plt.xlim(ax1.properties.XLim)
	plt.yticks(yticks, range(1, len(yticks) + 1))

	folder = "/home/alex/r"
	title_for_file = '_'.join(title.split())
	plt.savefig(f"{folder}/{title_for_file}_{rat.replace('.fig', '')}.png", format="png", dpi=200)
	plt.close()
