import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

step = 0.25
sim_time = 1000

# init random data for example

with open("/home/alex/MP_E.dat") as file:
	voltages1 = [float(v) for v in file.readline().split()]
with open("/home/alex/MP_F.dat") as file:
	voltages2 = [float(v) - 70 for v in file.readline().split()]

data1 = voltages1 # y
data2 = voltages2  # y
times = [t * step for t in range(len(data1))]  # x

xMin = 0
xMax = sim_time
yMin = -150
yMax = 150

main_pen = {'color': (0, 0, 0), 'width': 1}
slices_pen = {'color': (255, 0, 0, 200), 'width': 2, 'style': QtCore.Qt.DashLine}

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
plot_main.plot(times, data1, pen=main_pen)
plot_main.plot(times, data2, pen=main_pen)
plot_main.showGrid(x=True, y=True)
plot_main.setLabel('left', text="Voltage, mV")
plot_main.setLimits(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)

# add the RegionItem
zoom_borders = pg.LinearRegionItem([0, 25], brush=(100, 100, 100, 50))
pg.InfiniteLine(pos=2, pen={'color': (255, 0, 0, 200), 'width': 2})  # ??
zoom_borders.setZValue(-10)  # ??
plot_main.addItem(zoom_borders)

# move on the next row
win.nextRow()

# add the second plot
plot_zoomed = win.addPlot(title="Zoomed data")
# plotting and settings
plot_zoomed.plot(times, data1, pen=main_pen)
plot_zoomed.plot(times, data2, pen=main_pen)
plot_zoomed.showGrid(x=True, y=True)
plot_zoomed.setLabel('left', text="Voltage, mV")
plot_zoomed.setLabel('bottom', text="Time, ms")
plot_zoomed.setLimits(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)

# add slice times to each plot
for slice_time in range(0, sim_time):
	if slice_time in [125, 275, 400, 550, 675, 675+150, 675+150+125]:
		x = [slice_time, slice_time]
		plot_main.plot(x, [-100, 100], pen=slices_pen)
		plot_zoomed.plot(x, [-100, 100], pen=slices_pen)


# update plot after each changing of the region item
def update_plot():
	# change range the same as region value
	plot_zoomed.setXRange(*zoom_borders.getRegion(), padding=0)


# behavior of changing the region
def update_region():
	# change size of the region if we changed size of the zooming area
	zoom_borders.setRegion(plot_zoomed.getViewBox().viewRange()[0])


# add behavior for actions
zoom_borders.sigRegionChanged.connect(update_plot)
plot_zoomed.sigXRangeChanged.connect(update_region)

update_plot()

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()
