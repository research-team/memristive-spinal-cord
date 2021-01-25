import numpy as np
import RigolWFM.wfm as rigol
import matplotlib.pyplot as plt


def moving_average(xdata, weight):
	return np.convolve(xdata, np.ones(weight), 'valid') / weight


def read(filename):
	waves = rigol.Wfm.from_file(filename, 'DS1000Z')
	return waves.channels


def decode(filename):
	for chan in read(filename):
		print(chan)
		volts = (chan.volts + chan.volt_offset) / chan.volt_per_division
		if chan.channel_number == 2:
			volts = moving_average(volts, 50)
		xticks = np.arange(volts.size) * chan.seconds_per_point
		plt.plot(xticks, volts)
	plt.show()


if __name__ == '__main__':
	decode("/home/alex/NewFile1.wfm")
