"""
The script to read mat files of the rats brains and mark peaks.
KDE analysis of the peaks.
"""

import logging
import scipy.io
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objects as go

from itertools import chain
from scipy.signal import argrelextrema

flatten = chain.from_iterable
Y_OFFSET = 1000

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

def load_data(filepath):
	"""
	Load data from the .mat format
	Args:
		filepath (str): path to the file
	Returns:
		dict: dictionary with variable names as keys, and loaded matrices as values
	"""
	mat = scipy.io.loadmat(filepath)
	log.info(f"File ({filepath}) loaded")
	return mat

def find_extrema(array, condition):
	"""
	Advanced wrapper of numpy.argrelextrema
	Args:
		array (np.ndarray): data array
		condition (np.ufunc): e.g. np.less (<), np.great_equal (>=) and etc.
	Returns:
		np.ndarray: indexes of extrema
		np.ndarray: values of extrema
	"""
	# get indexes of extrema
	indexes = argrelextrema(array, condition)[0]
	# in case where data line is horisontal and doesn't have any extrema -- return None
	if len(indexes) == 0:
		return None, None
	# get values based on found indexes
	values = array[indexes]
	# calc the difference between nearby extrema values
	diff_nearby_extrema = np.abs(np.diff(values, n=1))
	# form indexes where no twin extrema (the case when data line is horisontal and have two extrema on borders)
	indexes = np.array([index for index, diff in zip(indexes, diff_nearby_extrema) if diff > 0] + [indexes[-1]])
	# get values based on filtered indexes
	values = array[indexes]

	return indexes, values

def find_peaks(channels_data, dstep, border_time, border_ampl, debug=False):
	"""
	Function for extrema (peaks) finding
	Args:
		channels_data (np.ndarray): 2D array (channel, channel data)
		dstep (float): data step size
		border_time (list): the time border [min, max]
		border_ampl (list): the amplitude border [min, max]
		debug (bool): debug mode (plotting)
	Returns:
		list: peak times, amplitudes and channels
	"""
	channels_num, _ = channels_data.shape
	# generate 2D list by channels number
	peaks_time = [[] for _ in range(channels_num)]
	peaks_ampl = [[] for _ in range(channels_num)]
	peaks_chan = [[] for _ in range(channels_num)]

	for index, channel in enumerate(channels_data):
		# combine slices into one myogram
		e_max_inds, e_max_vals = find_extrema(channel, np.greater)
		e_min_inds, e_min_vals = find_extrema(channel, np.less)

		# start pairing extrema from maxima
		offset = slice(1, None) if e_min_inds[0] < e_max_inds[0] else slice(None)
		comb = list(zip(e_max_inds, e_min_inds[offset]))

		# create list for debugging (plot max values of peaks)
		max_value_peaks = []
		# process each extrema pair
		for max_index, min_index in comb:
			max_value = e_max_vals[e_max_inds == max_index][0]
			min_value = e_min_vals[e_min_inds == min_index][0]
			dT = abs(max_index - min_index) * dstep
			dA = abs(max_value - min_value)
			# check the difference between maxima and minima
			if (border_time[0] <= dT <= border_time[1]) and border_ampl[0] <= dA <= border_ampl[1]:
				peaks_time[index].append(max_index)
				peaks_ampl[index].append(dA)
				peaks_chan[index].append(index)
				max_value_peaks.append(max_value)
		#
		if debug:
			xticks = np.arange(len(channel)) * dstep
			y_offset = index * Y_OFFSET
			# plot the curve
			plt.plot(xticks, channel + y_offset, color='k')
			# plot the extrema
			plt.plot(e_max_inds * dstep, e_max_vals + y_offset, '.', color='r')
			plt.plot(e_min_inds * dstep, e_min_vals + y_offset, '.', color='b')
			# plot the peaks
			x = np.asarray(peaks_time[index]) * dstep
			y = np.asarray(max_value_peaks) + np.asarray(peaks_chan[index]) * Y_OFFSET
			plt.plot(x, y, '.', color='g', ms=20)

	if debug:
		plt.show()

	return np.asarray(peaks_time), np.asarray(peaks_ampl), np.asarray(peaks_chan)


def smooth_data(channels, box_pts):
	"""
	Smoothing a curves
	Args:
		channels (np.ndarray): 2D array, y-axis data
		box_pts (int):
	Returns:
		np.ndarray: smoothed aata
	"""
	box = np.ones(box_pts) / box_pts
	for channel in channels:
		channel[:] = np.convolve(channel, box, mode='same')

	return channels

def get_channels(data, name, channels=None, dstep=0.025, debug=False):
	"""
	Get the data from the dict mat file
	Args:
		data (dict): dict of the mat file
		name (str): name of the channels?
		channels (list): a range list of channels for extracting
		dstep (float): data step size
		debug (bool): debug mode (plotting)
	Returns:
		np.ndarray: 2D array (channel, values)
	"""
	extracted = data.get(name, None)
	if extracted is None:
		raise KeyError(f"The key '{name}' does not exist!")

	log.info(f"data shape {name} {extracted.shape}")

	# re-order to 1. experiment; 2. channel; 3. channel data
	extracted = np.moveaxis(extracted, [0, 1, 2], [2, 1, 0])
	exp_num, cha_num, val_num = extracted.shape

	# FIXME get the channels of the 30th experiment?!
	channels = slice(*channels)
	channels_data = extracted[30, channels, :]

	if debug:
		exp_time = val_num * dstep
		fig, ax = plt.subplots(figsize=(30, 7))
		ax.set_title("Channels")
		for index, channel in enumerate(channels_data):
			xticks = np.linspace(0, exp_time, val_num)
			ax.plot(xticks, channel + index * Y_OFFSET, lw=1)
		plt.show()

	log.info("Shape of the channels")
	log.info(f"Shape {channels_data.shape}")

	return channels_data

def plot_3D_density(X, Y, xmin, xmax, ymin, ymax, factor=8, filepath=None):
	"""
	Plots the 3D density graphics
	Args:
		X (np.ndarray): flatten 1D array
		Y (np.ndarray): flatten 1D array
		xmin (float or int): minimal X data value
		xmax (float or int): maximal X data value
		ymin (float or int): minimal Y data value
		ymax (float or int): maximal Y data value
		factor (float or int): gridsize factor
		filepath (str): filepath for .html saving
	"""
	if filepath is None:
		return
	# form a mesh grid
	gridsize_x, gridsize_y = factor * xmax, factor * ymax
	xmesh, ymesh = np.meshgrid(np.linspace(xmin, xmax, gridsize_x),
	                           np.linspace(ymin, ymax, gridsize_y))
	xmesh = xmesh.T
	ymesh = ymesh.T
	# re-present grid in 1D and pair them as (x1, y1 ...)
	positions = np.vstack([xmesh.ravel(), ymesh.ravel()])
	values = np.vstack((X, Y))
	# use a Gaussian KDE
	a = st.gaussian_kde(values)(positions).T
	# re-present grid back to 2D
	z = np.reshape(a, xmesh.shape)

	z_contours = {"show": True,
	              "start": np.min(z) - 0.00001,
	              "end": np.max(z) + 0.00001,
	              "size": (np.max(z) - np.min(z)) / 16,
	              'width': 1,
	              "color": "gray"}
	surface = go.Surface(x=xmesh, y=ymesh, z=z, contours=dict(z=z_contours), opacity=1)

	# plot the 3D
	fig = go.Figure(data=surface)
	# change a camera view and etc
	fig.update_layout(title=f'test title', width=1000, height=800, autosize=False,
	                  scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.25, y=-1.25, z=1.25)),
	                  scene=dict(xaxis=dict(title_text="Time, ms",
	                                        titlefont=dict(size=30),
	                                        ticktext=list(range(26))),
	                             yaxis=dict(title_text="Channel №",
	                                        titlefont=dict(size=30),
	                                        tickvals=list(range(ymax + 1)),
	                                        ticktext=list(range(1, ymax + 2))),
	                             aspectratio={"x": 1, "y": 1, "z": 0.5}))

	py.plot(fig, validate=False, filename=f"{filepath}/test.html", auto_open=True)


def main():
	# file_prefix = '../../data/rats/'
	# file_path = '2011_05_03_0011.mat'
	# file_path = '2011_05_03_0023.mat'
	file_prefix = "/home/alex/Downloads/Telegram Desktop"
	filename = '2011_05_03_0003.mat'
	filepath = f"{file_prefix}/{filename}"

	dstep = 0.025   # TODO is it true?
	# choose the channels range
	channels = [0, 15]
	# min/max
	border_time = [0.1, 3]
	border_ampl = [100, np.inf]

	data = load_data(filepath)
	channel_data = get_channels(data, channels=channels, name='lfp', debug=True)
	channel_data = smooth_data(channel_data, 20)

	peaks_time, peaks_ampl, peaks_chan = find_peaks(channel_data, dstep, border_time, border_ampl, debug=True)

	# choose the data
	x_data = np.array(list(flatten(peaks_time))) * dstep
	y_data = np.array(list(flatten(peaks_chan)))
	z_data = np.array(list(flatten(peaks_ampl)))

	plot_3D_density(x_data, y_data, xmin=0, xmax=50, ymin=0, ymax=15, filepath=file_prefix)

if __name__ == "__main__":
	main()
