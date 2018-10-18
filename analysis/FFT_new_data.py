import logging
import csv
from analysis.FFT import fast_fourier_transform
import pylab as plt
import numpy
logging.basicConfig(level=logging.DEBUG)
with open('../bio-data//4_Rat-16_5-09-2017_RMG_one_step_T.txt') as inf:
    reader = csv.reader(inf, delimiter='\t')
    data = list(zip(*reader))[2]
RMG_column = []
volt_data = []
for i in range (6, len(data)):
    if data[i] != 'NaN':
        RMG_column.append(data[i])
# print(len(RMG_column), "RMG_column", RMG_column)
for i in range(len(RMG_column)):
    volt_data.append(float(RMG_column[i]))
# print(len(volt_data), "volt_data", volt_data)
sliced_data = volt_data[0:100]
length = len(volt_data)
FD = 4000
frequency = fast_fourier_transform(volt_data)
plt.title("RMG 9m min one step")
plt.plot(volt_data)
plt.show()
plt.plot(frequency)
plt.show()
a = numpy.fft.rfftfreq(length, 1. / FD)
# for i in range(len(a)):
# 	logging.debug(a[i])
# logging.debug(len(numpy.fft.rfftfreq((length ) - 1, 1. / FD)))
# logging.debug(len(numpy.abs(frequency)))
# logging.debug(len(frequency))
plt.xlim(0, 500)
plt.title("4_Rat-16_5-09-2017_RMG_one_step_T")
plt.plot(numpy.fft.fftfreq(length, 1. / FD), numpy.abs(frequency))
plt.show()
