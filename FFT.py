import numpy
from analysis.real_data_slices import read_data, slice_myogram
import os
import pylab as plt
import logging 
#http://old.pynsk.ru/posts/2015/Nov/09/matematika-v-python-preobrazovanie-fure/#.W8RQmHszZhE
logging.basicConfig(level=logging.DEBUG)


def fast_fourier_transform(volt_data):
	"""

	Parameters
	----------
	volt_data: list
		the voltages array (raw_real_data processed with slice_myogram fuction (first returned list))

	Returns
	-------
	four_tran: list
		the result of the work of numpy.fft.fft function, array, processed with fast Fourier transform
	"""
	four_tran = numpy.fft.fft(volt_data)
	# for i in range(len(four_tran)):
	# 	logging.debug(four_tran[i])
	# logging.debug(len(four_tran))
	return four_tran


raw_real_data = read_data(os.getcwd() + '/../bio-data/SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
myogram_data = slice_myogram(raw_real_data)
volt_data = myogram_data[0]
sliced_data = volt_data[0:100]
# for i in range(slices_begin_time)
length = len(sliced_data)
logging.debug("length = ", length)
FD = 400
frequency = fast_fourier_transform(sliced_data)
# plt.plot(volt_data)
# plt.show()
# plt.plot(frequency)
# plt.show()
# a = numpy.fft.rfftfreq(length, 1. / FD)
# for i in range(len(a)):
# 	logging.debug(a[i])
# logging.debug(len(numpy.fft.rfftfreq((length ) - 1, 1. / FD)))
# logging.debug(len(numpy.abs(frequency)))
# logging.debug(len(frequency))
plt.plot(numpy.fft.fftfreq(length, 1. / FD), numpy.abs(frequency))
plt.show()
