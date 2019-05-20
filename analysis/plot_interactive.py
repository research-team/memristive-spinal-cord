import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from analysis.patterns_in_bio_data import bio_slices, bio_data_runs
step = 0.25
sim_time = 100

# init random data for example

data1 = bio_slices()# y
print(data1)

xMin = 0
xMax = sim_time
yMin = -1
yMax = 150

main_pen = {'color': (0, 0, 0), 'width': 1}

# create the app
app = QtGui.QApplication([])
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

# add a window
win = pg.GraphicsWindow()
win.resize(1000, 600)
win.setBackground('#fff')
win.setWindowTitle("Flexor / Extensor visualization")

# add the first plot
plot_main = win.addPlot()
# plotting and settings
colors = ['black', 'rosybrown', 'firebrick', 'sandybrown', 'gold', 'olivedrab', # 6
          'darksalmon', 'green', 'sienna', 'darkblue', 'coral', 'orange',   # 12
          'darkkhaki', 'red', 'tan', 'steelblue', 'darkgreen', 'darkblue', 'palegreen', 'k', 'forestgreen',   # 21
          'slategray', 'limegreen', 'dimgrey', 'darkorange', 'darkgreen', 'cornflowerblue', 'dimgray', 'burlywood', # 29
          'royalblue', 'grey', 'g', 'gray', 'lime', 'midnightblue', 'seagreen', 'navy']   # 37
yticks = []
color_number = 0
for index, sl in enumerate(data1):
	offset = index * 5
	times = [time * step for time in range(len(data1[0][0]))]
	for run in range(len(sl)):
		plot_main.plot(times, [s + offset for s in sl[run]], color=colors[color_number], linewidth=1)
	color_number += 1
	yticks.append(sl[run][0] + offset)
plot_main.showGrid(x=True, y=True)
plot_main.setLimits(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)

# move on the next row
win.nextRow()

# add slice times to each plot
for slice_time in range(0, sim_time):
	if slice_time in [125, 275, 400, 550, 675, 675+150, 675+150+125]:
		x = [slice_time, slice_time]
		plot_main.plot(x, [-100, 100])


# update plot after each changing of the region item
# def update_plot():
	# change range the same as region value
	# plot_zoomed.setXRange(*zoom_borders.getRegion(), padding=0)


# behavior of changing the region
# def update_region():
	# change size of the region if we changed size of the zooming area
	# zoom_borders.setRegion(plot_zoomed.getViewBox().viewRange()[0])


# add behavior for actions
# zoom_borders.sigRegionChanged.connect(update_plot)
# plot_zoomed.sigXRangeChanged.connect(update_region)

# update_plot()

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()