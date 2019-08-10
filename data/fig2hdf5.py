import numpy as np
import pylab as plt
from scipy.io import loadmat


def plot_fig(filename):
	d = loadmat(filename, squeeze_me=True, struct_as_record=False)
	ax1 = d['hgS_070000'].children

	if np.size(ax1) > 1:
		ax1 = ax1[0]

	plt.figure(figsize=(16, 9))

	for line in ax1.children:
		if line.type == 'graph2d.lineseries':
			mark = '.'
			linestyle = '-'
			r, g, b = 0, 0, 1
			marker_size = 1

			if hasattr(line.properties, 'Marker'):
				mark = str(line.properties.Marker)[0]
			if hasattr(line.properties, 'LineStyle'):
				linestyle = str(line.properties.LineStyle)
			if hasattr(line.properties, 'Color'):
				r, g, b = line.properties.Color
			if hasattr(line.properties, 'MarkerSize'):
				marker_size = line.properties.MarkerSize

			x = line.properties.XData
			y = line.properties.YData

			plt.plot(x, y, marker=mark, linestyle=linestyle, color=(r, g, b), markersize=marker_size)


	plt.xlim(ax1.properties.XLim)

	plt.show()


path = '/home/alex/GitHub/data/spinal/QPZ/extensor/13.5cms/sliced/7/#7_112309_QUIP_BIPEDAL_burst7_Ton_21.fig'
plot_fig(path)

