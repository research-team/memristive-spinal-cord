import scipy.signal as signal
import scipy.io as sio
import numpy as np
datas = {}
mat_data = sio.loadmat('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
# Collect data
for index, data_title in enumerate(mat_data['titles']):
	data_start = int(mat_data['datastart'][index])-1
	data_end = int(mat_data['dataend'][index])
	if "Stim" not in data_title:
		datas[data_title] = mat_data['data'][0][data_start:data_end]
window = signal.general_gaussian(51, p=0.5, sig=20)
filtered = signal.fftconvolve(window, datas)
filtered = (np.average(datas) / np.average()) * filtered
filtered = np.roll(filtered, -25 )
peaks = signal.find_peaks_cwt(filtered, np.arange(100, 200))
print(peaks)