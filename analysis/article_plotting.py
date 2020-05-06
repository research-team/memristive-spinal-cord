import os
import pickle
import logging
import numpy as np
import pylab as plt
import scipy.stats as st
import plotly.offline as py
import plotly.graph_objects as go
import matplotlib.patches as mpatches

from colour import Color
from itertools import chain
from matplotlib import gridspec
from scipy.stats import ks_2samp
from scipy.signal import argrelextrema
from rpy2.robjects.packages import STAP
from scipy.stats import kstwobign, anderson_ksamp
from matplotlib.ticker import MaxNLocator, MultipleLocator
from analysis.functions import read_hdf5, trim_data, calibrate_data, get_boxplots, parse_filename, subsampling


class Analyzer:
	def __init__(self, pickle_folder, debug=False):
		self.pickle_folder = pickle_folder
		self.plots_folder = f"{pickle_folder}/plots"

		self.debug = debug

		if not os.path.exists(self.pickle_folder):
			os.makedirs(self.pickle_folder)
		if not os.path.exists(self.plots_folder):
			os.makedirs(self.plots_folder)

		self.flatten = chain.from_iterable

		logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
		self.log = logging.getLogger()

	@staticmethod
	def disable_log():
		logging.disable(logging.CRITICAL)

	def get_pickle_data(self, file):
		with open(f"{self.pickle_folder}/{file}.pickle", 'rb') as handle:
			pdata = pickle.load(handle)
		return pdata

	def get_rats_id(self, pdata, muscle='E'):
		return pdata['rats_data'][muscle].keys()

	def get_myograms(self, pdata, rat, muscle='E'):
		return pdata['rats_data'][muscle][rat]['data']

	def get_peak_times(self, pdata, rat, muscle='E', flatten=False):
		data = pdata['rats_data'][muscle][rat]['times']
		if flatten:
			return np.array(list(self.flatten(self.flatten(data))))
		return data

	def get_peak_ampls(self, pdata, rat, muscle='E', flatten=False):
		data = pdata['rats_data'][muscle][rat]['ampls']
		if flatten:
			return np.array(list(self.flatten(self.flatten(data))))
		return data

	def get_peak_slices(self, pdata, rat, muscle='E', flatten=False):
		data = pdata['rats_data'][muscle][rat]['slices']
		if flatten:
			return np.array(list(self.flatten(self.flatten(data))))
		return data

	def get_shortname(self, pdata):
		return pdata['shortname']

	def get_dstep_to(self, pdata):
		return pdata['dstep_to']

	def get_dstep_from(self, pdata):
		return pdata['dstep_from']

	def get_latency_volume(self, file, rats):
		return self.plot_density_3D(file, rats, factor=15, only_volume=True)

	@staticmethod
	def _recolor(boxplot_elements, color, fill_color, fill_alpha=0.0):
		"""
		Add colors to bars (setup each element)
		Args:
			boxplot_elements (dict): components of the boxplot
			color (str): HEX color of outside lines
			fill_color (str): HEX color of filling
		"""
		assert 0 <= fill_alpha <= 1

		hex_alpha = hex(int(fill_alpha * 255))[2:]
		# TODO убрать plt, сделать привязку к axes
		for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
			plt.setp(boxplot_elements[element], color=color, linewidth=2)
		plt.setp(boxplot_elements["fliers"], markeredgecolor=color)
		for patch in boxplot_elements['boxes']:
			patch.set(facecolor=f"{fill_color}{hex_alpha}")

	def _lw_prepare_data(self, folder, muscle, metadata, fill_zeros):
		"""

		Args:
			muscle:
			metadata:
		"""
		# constant number of slices (mode, speed): (extensor, flexor))
		slices_number_dict = {
			("PLT", '21'): (6, 5),
			("PLT", '13.5'): (12, 5),
			("PLT", '6'): (30, 5),
			("TOE", '21'): (4, 4),
			("TOE", '13.5'): (8, 4),
			("AIR", '13.5'): (5, 4),
			("QPZ", '13.5'): (12, 5),
			("STR", '21'): (6, 5),
			("STR", '13.5'): (12, 5),
			("STR", '6'): (30, 5),
		}
		#
		dstep_from = metadata['dstep_from']
		filename = metadata['filename']
		dstep_to = metadata['dstep_to']
		source = metadata['source']
		speed = metadata['speed']
		mode = metadata['mode']
		slise_in_ms = 1 / int(metadata['rate']) * 1000

		if muscle == 'F':
			filename = filename.replace('_E_', '_F_')

		standard_slice_length_in_steps = int(25 / dstep_to)
		e_slices_number, f_slices_number = slices_number_dict[(mode, speed)]
		#
		for rat_id in range(10):
			abs_filename = f"{folder}/{filename}"
			# read the raw data
			dataset = read_hdf5(abs_filename, rat=rat_id)

			if dataset is None:
				continue

			dataset = subsampling(dataset, dstep_from=dstep_from, dstep_to=dstep_to)
			if source == "bio":
				# get the size of the data which we are waiting for
				if muscle == "E":
					full_size = int(e_slices_number * 25 / dstep_to)
				else:
					full_size = int(f_slices_number * 25 / dstep_to)
				#
				if fill_zeros:
					prepared_data = np.array([d + [0] * (full_size - len(d)) for d in calibrate_data(dataset, source)])
				else:
					prepared_data = np.array([d for d in calibrate_data(dataset, source) if len(d) == full_size])
			else:
				# extract data of extensor
				if muscle == "E":
					begin = 0
					end = begin + e_slices_number * standard_slice_length_in_steps
				# extract data of flexor
				else:
					begin = standard_slice_length_in_steps * e_slices_number
					end = begin + (7 if metadata['pedal'] == "4" else 5) * standard_slice_length_in_steps
				# trim redundant simulation data
				dataset = trim_data(dataset, begin, end)
				prepared_data = np.array(calibrate_data(dataset, filename))

			if muscle == "E":
				sliced_data = [np.array_split(beg, e_slices_number) for beg in prepared_data]
			else:
				sliced_data = [np.array_split(beg, f_slices_number) for beg in prepared_data]

			sliced_data = np.array(sliced_data)

			sliced_time, sliced_ampls, sliced_index = self._get_peaks(sliced_data, dstep_to, [0, slise_in_ms], debug=self.debug)
			metadata['rats_data'][muscle][rat_id] = dict(data=sliced_data,
			                                             times=sliced_time,
			                                             ampls=sliced_ampls,
			                                             slices=sliced_index)

	def prepare_data(self, folder, dstep_to=None, fill_zeros=True):
		"""

		Args:
			folder:
			dstep_to:
			fill_zeros:
		"""
		# check each .hdf5 file in the folder
		for filename in [f for f in os.listdir(folder) if f.endswith('.hdf5') and '_E_' in f]:
			source, muscle, mode, speed, rate, pedal, dstep = parse_filename(filename)
			shortname = f"{source}_{mode}_{speed}_{pedal}ped"
			#
			if dstep_to is None:
				dstep_to = dstep
			# prepare metadata dict for fillining
			metadata = {
				'filename': filename,
				'source': source,
				'muscle': muscle,
				'mode': mode,
				'speed': speed,
				'rate': rate,
				'pedal': pedal,
				'dstep_from': dstep,
				'dstep_to': dstep_to,
				'shortname': shortname,
				'rats_data': {
					'E': {},
					'F': {}
				}
			}

			self._lw_prepare_data(folder, 'E', metadata, fill_zeros)
			self._lw_prepare_data(folder, 'F', metadata, fill_zeros)

			pickle_save = f"{self.pickle_folder}/{os.path.splitext(filename)[0]}.pickle"
			with open(pickle_save, 'wb') as handle:
				pickle.dump(metadata, handle)
			logging.info(pickle_save)

			del metadata

	@staticmethod
	def _form_ticklabels(ticks_length):
		"""
		Form a ticklabels there are 4 ticks: begin, 1/3, 2/3 and the end
		Args:
			ticks_length: 
		Returns:
			list: prepared ticklabels
		"""
		ytick_labels = [None] * ticks_length
		yticks_indices = range(1, ticks_length + 1)
		for i in [0, -1, int(1 / 3 * ticks_length), int(2 / 3 * ticks_length)]:
			ytick_labels[i] = yticks_indices[i]
		return ytick_labels

	@staticmethod
	def rat_metadata(rat_steps, flatten_metadata):
		return len(flatten_metadata) / rat_steps, np.median(flatten_metadata)

	def plot_fMEP_boxplots(self, file, rats=None, borders=(0, 25), show=False):
		"""
		Plot slices with peaks and their boxplots inside borders
		Args:
			file:
			rats:
			borders:
			show:
		"""
		muscle = 'E'
		pdata = self.get_pickle_data(file)

		if rats is None or rats is all:
			rats = self.get_rats_id(pdata)
		if rats is int:
			rats = [rats]
		speed = pdata['speed']
		dstep_to = pdata['dstep_to']
		shortname = pdata['shortname']

		# plot per each rat
		for rat_id in rats:
			rat_myogram = self.get_myograms(pdata, rat_id, muscle=muscle)
			rat_times = self.get_peak_times(pdata, rat_id, muscle=muscle)

			total_steps = rat_myogram.shape[0]
			total_slices = rat_myogram.shape[1]

			sliced_x = [[] for _ in range(total_slices)]
			sliced_y = [[] for _ in range(total_slices)]
			passed = 0
			alles = total_steps * total_slices

			plt.close()
			if speed == "6":
				fig, ax = plt.subplots(figsize=(20, 20))
			elif speed == "13.5":
				fig, ax = plt.subplots(figsize=(16, 12))
			else:
				fig, ax = plt.subplots(figsize=(16, 8))

			save_filename = f"{shortname}_{rat_id}_fMEP_boxplot"

			colors = iter(["#275b78", "#f2aa2e", "#a6261d", "#472650"] * total_steps)
			for data_per_step, time_per_step in zip(rat_myogram, rat_times):
				color = next(colors)
				for slice_index, (slice_data, slice_times) in enumerate(zip(data_per_step, time_per_step)):
					myogram = np.array(slice_data) + slice_index
					peaks_time = np.array(slice_times)
					plt.plot(np.arange(len(myogram)) * dstep_to, myogram, alpha=0.5, color=color)
					peaks_time = peaks_time[(peaks_time >= borders[0] / dstep_to) & (peaks_time <= borders[1] / dstep_to)]
					if len(peaks_time) > 0:
						passed += 1
						sliced_x[slice_index] += list(peaks_time)
						sliced_y[slice_index] += list(myogram[peaks_time])
						plt.plot(peaks_time * dstep_to, myogram[peaks_time], '.', c='k')

			for i, (x, y) in enumerate(zip(sliced_x, sliced_y)):
				if len(x):
					x = np.array(x)
					plt.plot(x * dstep_to, y, '.', color='k', ms=4)
					bx = plt.boxplot(x * dstep_to, vert=False, positions=[i], widths=0.8,
					                 showfliers=False, patch_artist=True, whis=[10, 90])
					starts = bx['whiskers'][0].get_xdata()[1]
					# starts = np.percentile(s * dstep_to, percentile)[0]
					plt.text(x=starts - 1.5, y=i + 0.2, s=f"{starts:.1f}", fontsize=25)
					self._recolor(bx, "#287a72", "#287a72", fill_alpha=0.2)

			plt.grid(which='both', axis='x')
			self.axis_article_style(ax, axis='x')

			plt.yticks(range(0, total_slices), self._form_ticklabels(total_slices), fontsize=30)
			plt.xlim(0, 25)
			plt.savefig(f"{self.plots_folder}/{save_filename}.pdf", format="pdf")
			if show:
				plt.show()
			plt.close()
			logging.info(f"{shortname}, rat {rat_id}, {passed / alles * 100:.1f}% of peaks prob. at {borders}ms")


	def plot_density_3D(self, file, rats=None, factor=8, show=False, only_volume=False):
		"""

		Args:
			file:
			rats:
			factor:
			show:
			only_volume:
		Returns:
		"""
		pdata = self.get_pickle_data(file)

		if rats is None or rats is all:
			rats = self.get_rats_id(pdata)
		if rats is int:
			rats = [rats]

		volumes = []
		shortname = self.get_shortname(pdata)
		dstep = self.get_dstep_to(pdata)

		for rid in rats:
			X = self.get_peak_times(pdata, rid, muscle='E', flatten=True) * dstep
			Y = self.get_peak_slices(pdata, rid, muscle='E', flatten=True)

			save_filename = f"{shortname}_3D_rat={rid}"
			# form a mesh grid
			xmax, ymax = 25, max(Y)
			xborder_l, xborder_r = 5, 20

			gridsize_x, gridsize_y = factor * xmax, factor * ymax
			xmesh, ymesh = np.meshgrid(np.linspace(0, xmax, gridsize_x),
			                           np.linspace(0, ymax, gridsize_y))
			xmesh = xmesh.T
			ymesh = ymesh.T
			# re-present grid in 1D and pair them as (x1, y1 ...)
			positions = np.vstack([xmesh.ravel(), ymesh.ravel()])
			values = np.vstack((X, Y))
			# use a Gaussian KDE
			a = st.gaussian_kde(values)(positions).T
			# re-present grid back to 2D
			z = np.reshape(a, xmesh.shape)
			# set the mid isoline (2/3)
			z_mid = (np.max(z) + np.min(z)) / 3 * 2

			conty_ymax = -np.inf
			conty_ymin = np.inf

			mid_contours = plt.contour(xmesh, ymesh, z, levels=[z_mid], alpha=0).allsegs[0]
			for contour in mid_contours:
				for x, y in contour:
					if xborder_l <= x <= xborder_r:
						if y > conty_ymax:
							conty_ymax = y
						if y < conty_ymin:
							conty_ymin = y
			# clip data by time [5, 20] and slices [contour ymin, contour ymax]
			xslice = slice(xborder_l * factor, xborder_r * factor)
			yslice = slice(int(round(conty_ymin * factor)), int(round(conty_ymax * factor)))
			zclip = z[xslice, yslice]
			# filter clipped data that lower than 2/3 isoline
			zunder = zclip[zclip <= z_mid]
			# calculate a volune
			cellsize = xmesh[1][0] * ymesh[0][1]
			zvol = np.sum(np.abs(z_mid - zunder)) * cellsize

			if only_volume:
				volumes.append(zvol)
				continue

			surface = go.Surface(contours=dict(z={"show": True,
			                                      "start": np.min(z) - 0.00001,
			                                      "end": np.max(z) + 0.00001,
			                                      "size": (np.max(z) - np.min(z)) / 16,
			                                      'width': 1,
			                                      "color": "gray"}),
			                     x=xmesh,
			                     y=ymesh,
			                     z=z,
			                     opacity=1)

			# left plane of time border [5]
			plane1 = go.Surface(x=[xborder_l, xborder_l],
			                    y=[0, ymax],
			                    z=[[np.min(z), np.max(z)], [np.min(z), np.max(z)]],
			                    showscale=False, surfacecolor=[0] * 4, opacity=0.7, cmax=1, cmin=0)
			# right plane of time border [20]
			plane2 = go.Surface(x=[xborder_r, xborder_r],
			                    y=[0, ymax],
			                    z=[[np.min(z), np.max(z)], [np.min(z), np.max(z)]],
			                    showscale=False, surfacecolor=[0] * 4, opacity=0.7, cmax=1, cmin=0)
			# bottom plane of slice border
			plane3 = go.Surface(x=[xborder_l, xborder_r],
			                    y=[conty_ymin, conty_ymin],
			                    z=[[np.min(z), np.min(z)], [np.max(z), np.max(z)]],
			                    showscale=False, surfacecolor=[0] * 4, opacity=0.7, cmax=1, cmin=0)
			# top plane of slice border
			plane4 = go.Surface(x=[xborder_l, xborder_r],
			                    y=[conty_ymax, conty_ymax],
			                    z=[[np.min(z), np.min(z)], [np.max(z), np.max(z)]],
			                    showscale=False, surfacecolor=[0] * 4, opacity=0.7, cmax=1, cmin=0)
			# form data pack to visualize all in one axes
			data = [surface, plane1, plane2, plane3, plane4]
			# plot isoline
			for c in mid_contours:
				data.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=[z_mid] * len(c[:, 0]),
				                         line=dict(color='#000000', width=6), mode='lines', showlegend=False))
			# plot Z dots
			xunder = xmesh[xslice, yslice][zclip <= z_mid].flatten()
			yunder = ymesh[xslice, yslice][zclip <= z_mid].flatten()
			zunder = zunder.flatten()
			data.append(go.Scatter3d(x=xunder, y=yunder, z=zunder,
			                         mode='markers', marker=dict(size=2, color=['rgb(0,0,0)'] * len(zunder))))
			# plot all
			fig = go.Figure(data=data)
			# change a camera view and etc
			fig.update_layout(title=f'{shortname} | RAT {rid} | V: {zvol:.3f}',
			                  autosize=False,
			                  width=1000,
			                  height=800,
			                  scene_camera=dict(
				                  up=dict(x=0, y=0, z=1),
				                  eye=dict(x=-1.25, y=-1.25, z=1.25)
			                  ),
			                  scene=dict(
				                  xaxis=dict(title_text="Time, ms",
				                             titlefont=dict(size=30),
				                             ticktext=list(range(26))),
				                  yaxis=dict(title_text="Slices",
				                             titlefont=dict(size=30),
				                             tickvals=list(range(ymax + 1)),
				                             ticktext=list(range(1, ymax + 2))),
				                  aspectratio={"x": 1, "y": 1, "z": 0.5}
			                  ))
			py.plot(fig, validate=False,
			        filename=f"{self.plots_folder}/{save_filename}.html",
			        auto_open=show)
		if only_volume:
			return volumes

	def plot_kde_peaks_slices(self, peaks_times_pack, peaks_slices_pack, names, colors, packs_size, save_to):
		"""
		ToDo add info
		Args:
			peaks_times_pack (list): pack of peaks
			peaks_slices_pack (list): pack of slices number
			names (list): filenames for different origin of data
			colors (list): hex colors for graphics
			save_to (str): save folder
		"""
		# unpack the data
		for x, y, color, name, pack_size in zip(peaks_times_pack, peaks_slices_pack, colors, names, packs_size):
			# x - peak time
			# y - slice index
			# shift slice index to be human readable [1...N]
			y += 1
			# define grid for subplots
			gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 3], wspace=0.3)
			fig = plt.figure(figsize=(16, 9))
			kde_ax = plt.subplot(gs[1, 0])
			kde_ax.spines['top'].set_visible(False)
			kde_ax.spines['right'].set_visible(False)

			# 2D joint plot
			borders = 0, 25, 1, max(y)
			contour_plot(x=x, y=y, color=color, ax=kde_ax, z_prev=[0], borders=borders, levels_num=15)
			joint_plot(x, y, kde_ax, gs, **{"color": color, "pos": 0}, borders=borders, with_boxplot=False)
			log_text = f"total peaks={len(x)}, packs={pack_size}, per pack={int(len(x) / pack_size)}"
			# kde_ax.text(0.05, 0.95, log_text, transform=kde_ax.transAxes, verticalalignment='top', color=color, fontsize=10)
			logging.info(log_text)

			kde_ax.set_xlabel("peak time (ms)")
			kde_ax.set_ylabel("slice index")
			kde_ax.set_xlim(borders[0], borders[1])
			kde_ax.set_ylim(borders[2], borders[3])

			plt.tight_layout()
			new_filename = short_name(name)
			plt.savefig(f"{save_to}/{new_filename}_kde_peak_slice.pdf", dpi=250, format="pdf")
			plt.show()
			plt.close(fig)

			logging.info(f"saved to {save_to}")

	def plot_shadow_slices(self, file, rats=None, only_extensor=False, add_kde=False, show=False):
		"""

		Args:
			file:
			rats:
			show:
		"""
		shadow_color = "#472650"
		kde_color = "#287a72"
		k_fliers_high, k_fliers_low = 5, 6

		pdata = self.get_pickle_data(file)

		if rats is None or rats is all:
			rats = self.get_rats_id(pdata)
		if rats is int:
			rats = [rats]

		shortname = self.get_shortname(pdata)
		dstep_to = self.get_dstep_to(pdata)
		speed = pdata['speed']
		slice_in_ms = 1 / int(pdata['rate']) * 1000

		if speed == "6":
			figsize = (20, 20)
		elif speed == "13.5":
			figsize = (16, 12)
		else:
			figsize = (16, 8)
		#
		for rat_id in rats:
			extensor_data = self.get_myograms(pdata, rat_id, muscle='E')
			# check rat's flexor, in some cases there are no data
			flexor_flag = rat_id in self.get_rats_id(pdata, muscle='F') and not only_extensor
			# get number of slices per muscle
			e_slices_number = len(extensor_data[0])
			steps_in_slice = len(extensor_data[0][0])
			# calc boxplots of original data ()
			e_boxplots = get_boxplots(extensor_data)
			# combine data into one list

			plt.close('all')
			fig, ax = plt.subplots(figsize=figsize)

			yticks = []
			shared_x = np.arange(steps_in_slice) * dstep_to
			# plot extensor
			for slice_index, data in enumerate(e_boxplots):
				# set ideal or median
				ideal_data = extensor_data[0][slice_index] + slice_index
				data += slice_index
				# fliers shadow
				ax.fill_between(shared_x,  data[:, k_fliers_high], data[:, k_fliers_low],
				                color=shadow_color, alpha=0.7, zorder=3)
				# ideal pattern
				ax.plot(shared_x, ideal_data, color='k', linewidth=1, zorder=4)
				yticks.append(ideal_data[0])

			if flexor_flag:
				flexor_data = self.get_myograms(pdata, rat_id, muscle='F')
				f_slices_number = len(flexor_data[0])
				f_boxplots = get_boxplots(flexor_data)
				# plot flexor
				for slice_index, data in enumerate(f_boxplots):
					# set ideal or median
					ideal_data = flexor_data[0][slice_index] + slice_index + e_slices_number + 1
					data += slice_index + e_slices_number + 1
					# fliers shadow
					ax.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low],
					                color=shadow_color, alpha=0.7, zorder=3)
					# ideal pattern
					ax.plot(shared_x, ideal_data, color='k', linewidth=1, zorder=4)
					yticks.append(ideal_data[0])

			if add_kde:
				x = self.get_peak_times(pdata, rat_id, muscle='E', flatten=True) * dstep_to
				y = self.get_peak_slices(pdata, rat_id, muscle='E', flatten=True)
				borders = 0, slice_in_ms, -1, e_slices_number + 1
				self._contour_plot(x=x, y=y, color=kde_color, ax=ax, z_prev=[0], borders=borders, levels_num=15)
			# form ticks
			self.axis_article_style(ax, axis='x')

			slices_number = e_slices_number
			if flexor_flag:
				slices_number += f_slices_number

			plt.yticks(yticks, self._form_ticklabels(slices_number), fontsize=30)
			plt.xlim(0, slice_in_ms)
			plt.tight_layout()
			save_filename = f"{shortname}_{rat_id}_sliced.pdf"

			plt.savefig(f"{self.plots_folder}/{save_filename}", dpi=250, format="pdf")
			if show:
				plt.show()
			plt.close()
			logging.info(f"{shortname}, rat {rat_id} "
			             f"{'' if flexor_flag else 'WITHOUT FLEXOR'} "
			             f"saved to {self.plots_folder}/{save_filename}")


	def plot_ks2d(self, peaks_times_pack, peaks_ampls_pack, names, colors, borders, packs_size, save_to,
	              additional_tests=False):
		raise NotImplemented
		"""
		ToDo add info
		Args:
			peaks_times_pack (list): pack of peak lists for different origin of data (bio/neuron/gras/nest)
			peaks_ampls_pack (list): pack of amplitude lists for different origin of data (bio/neuron/gras/nest)
			names (list): filenames for different origin of data
			colors (list): hex colors for graphics
			save_to (str): save folder
		"""
		# calc the critical constant to comapre with D-value * en
		crit = kstwobign.isf(0.01)
		# prepare filename
		new_filename = f"{'='.join(map(short_name, names))}"
		# # set new parameters for logging
		# reload(logging)
		# logging.basicConfig(filename=f"{save_to}/{new_filename}.log",
		#                     filemode='a',
		#                     format='%(message)s',
		#                     level=logging.DEBUG)

		# unpack the data
		xmin, xmax, ymin, ymax = borders
		x1, y1 = peaks_times_pack[0], peaks_ampls_pack[0]
		x2, y2 = peaks_times_pack[1], peaks_ampls_pack[1]
		assert len(x1) == len(y1) and len(x2) == len(y2)

		fig, axes = plt.subplots(nrows=2, ncols=3)
		for data1, data2, row in ([x1, x2, 0], [y1, y2, 1]):
			# plot the CDF (not scaled)
			axes[row, 0].set_title("not scaled CDF")
			label = f"STD: {np.std(data1):.3}, MED: {np.median(data1):.3}"
			axes[row, 0].plot(sorted(data1), np.linspace(0, 1, len(data1)), color=colors[0], label=label)
			label = f"STD: {np.std(data2):.3}, MED: {np.median(data2):.3}"
			axes[row, 0].plot(sorted(data2), np.linspace(0, 1, len(data2)), color=colors[1], label=label)
			if row == 0:
				axes[row, 0].set_xlabel("Time (ms)")
			else:
				axes[row, 0].set_xlabel("Amplitude")
			axes[row, 0].set_ylabel("Probability")
			axes[row, 0].legend(fontsize=9)

			# plot the CDF (scaled) with D-value line
			binEdges = np.hstack([-np.inf, np.sort(np.concatenate([data1, data2])), np.inf])
			binCounts1 = np.histogram(data1, binEdges)[0]
			binCounts2 = np.histogram(data2, binEdges)[0]
			sampleCDF1 = np.cumsum(binCounts1, dtype=float) / np.sum(binCounts1)
			sampleCDF2 = np.cumsum(binCounts2, dtype=float) / np.sum(binCounts2)
			deltaCDF = np.abs(sampleCDF1 - sampleCDF2)
			maxDind = np.argmax(deltaCDF)

			KSstatistic = np.max(deltaCDF)

			axes[row, 1].set_title("scaled CDF")
			axes[row, 1].plot(sampleCDF1)
			axes[row, 1].plot(sampleCDF2)
			axes[row, 1].plot([maxDind, maxDind], [sampleCDF1[maxDind], sampleCDF2[maxDind]], color='k', ls='--')
			axes[row, 1].set_xlabel("Dots")
			axes[row, 1].set_ylabel("Probability")

			# delta CDF
			deltaCDF = sampleCDF1 - sampleCDF2
			axes[row, 2].set_title(f"scaled CDF diff (D={KSstatistic:.5f})")
			axes[row, 2].axhline(y=max(deltaCDF), ls='--')
			axes[row, 2].axhline(y=min(deltaCDF), ls='--')
			axes[row, 2].plot(deltaCDF)
			axes[row, 2].set_xlabel("Dots")

		# plt.show()
		plt.close(fig)

		logging.info(f"N1 {len(x1)}")
		logging.info(f"N2 {len(x2)}")
		# calc the "en"
		en = np.sqrt(len(x1) * len(x2) / (len(x1) + len(x2)))

		# 1D peak times analysis
		dvalue, _ = ks_2samp(x1, x2)
		den = dvalue * en
		pvalue = kstwobign.sf(en * dvalue)
		logging.info("1D K-S peaks TIME")
		logging.info(f"Den ({den:.5f}) {'<' if den < crit else '>'} Critical ({crit:.5f})\n"
		             f"D-value: {dvalue:.5f}\n"
		             f"p-value: {pvalue}")
		logging.info("- " * 10)

		# 1D peak amplitudes analysis
		dvalue, _ = ks_2samp(y1, y2)
		en = np.sqrt(len(y1) * len(y2) / (len(y1) + len(y2)))
		den = dvalue * en
		pvalue = kstwobign.sf(den)
		logging.info("1D K-S peaks AMPL")
		logging.info(f"Den ({den:.5f}) {'<' if den < crit else '>'} Critical ({crit:.5f})\n"
		             f"D-value: {dvalue:.5f}\n"
		             f"p-value: {pvalue}")
		logging.info("- " * 10)
		colors = ['#4B244F', '#217B73']
		# define grid for subplots
		gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
		fig = plt.figure(figsize=(8, 6))
		kde_ax = plt.subplot(gs[1, 0])
		kde_ax.spines['top'].set_visible(False)
		kde_ax.spines['right'].set_visible(False)

		# 2D joint plot
		z_prev = np.zeros(1)
		label_pathes = []
		z = []
		kk = {'AIR 2pedal 13.5cms': 0.088,
		      'TOE 2pedal 13.5cms': 0.110,
		      'PLT 2pedal 6cms': 0.124,
		      'PLT 2pedal 21cms': 0.131,
		      'PLT 2pedal 13.5cms': 0.136,
		      'PLT 4pedal 21cms': 0.333,
		      'PLT 4pedal 13.5cms': 0.142,
		      'QPZ 2pedal 13.5cms': 0.199,
		      'STR 2pedal 21cms': 0.179,
		      'STR 2pedal 13.5cms': 0.210}

		for x, y, name, color, pack_size in zip(peaks_times_pack, peaks_ampls_pack, names, colors, packs_size):
			tmpname = name.split("_")
			k = kk[f"{tmpname[2]} {tmpname[5]} {tmpname[3]}"]
			z_prev = contour_plot(x=x, y=y, color=color, ax=kde_ax, z_prev=z_prev, borders=borders, levels_num=10)
			z.append(z_prev)
			t, r = joint_plot(x, y, kde_ax, gs, **{"color": color, 'k': k}, borders=borders, with_boxplot=False)
			# label_text = f"packs={pack_size}, per pack={int(len(x) / pack_size)}"
			# label_pathes.append(mpatches.Patch(color=color, label=label_text))

			axis_article_style(t, axis='x')
			axis_article_style(r, axis='y')

			t.set_xticklabels([])
			r.set_yticklabels([])

			t.set_xlim(borders[0], borders[1])
			r.set_ylim(borders[2], borders[3])

		axis_article_style(kde_ax, axis='both')

		# kde_ax.legend(handles=label_pathes, fontsize=17)
		kde_ax.set_xlim(borders[0], borders[1])
		kde_ax.set_ylim(borders[2], borders[3])

		plt.tight_layout()
		plt.savefig(f"{save_to}/{new_filename}_kde2d.pdf", dpi=250, format="pdf")
		# plt.show()
		plt.close(fig)

		logging.info(f"saved to {save_to}")

		if additional_tests:
			from colour import Color
			plt.close()
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			xmin, xmax, ymin, ymax = borders
			xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

			for zdata, clr in zip(z, colors):
				levels_num = 10
				h_norm, s_norm, l_norm = Color(clr).hsl
				# make an gradient based on number of levels
				light_gradient = np.linspace(l_norm, 0.95, levels_num)[::-1]
				# generate colors for contours level from HSL (normalized) to RGB (normalized)
				colors = [Color(hsl=(h_norm, s_norm, l_level)).rgb for l_level in light_gradient]
				# find an index of the maximal element
				m = np.amax(zdata)
				# form a step and levels
				step = (np.amax(zdata) - np.amin(zdata)) / levels_num
				levels = np.arange(0, m, step) + step

				ax.plot_surface(xx, yy, zdata, linewidth=0, alpha=0.2, color=clr)
				ax.plot_wireframe(xx, yy, zdata, linewidth=0.2, color=clr)

				ax.contour(xx, yy, zdata, zdir='z', offset=0, linewidths=0.5, levels=levels, colors=clr)
				ax.contourf(xx, yy, zdata, zdir='z', offset=0, levels=levels, colors=colors, alpha=0.5)

				xmaaa = np.max(zdata, axis=1)
				ymaaa = np.max(zdata, axis=0)
				ax.plot(np.linspace(xmin, xmax, len(xmaaa)), [1.5] * len(xmaaa), xmaaa, color=clr)
				ax.plot([8] * len(ymaaa), np.linspace(ymin, ymax, len(ymaaa)), ymaaa, color=clr)

			plt.show()

	@staticmethod
	def axis_article_style(ax, axis='both', auto_nbins=False, xshift=None, xmin=None, xmax=None):
		"""
		ToDo add info
		Args:
			ax (matplotlib.axes): currect figure axes
			axis (str): which axes change, both -- all
			auto_nbins (bool):
			xshift (None or float): offset of xticklabels
			xmin (None or float): set xlim for minimum
			xmax (None or float): set xlim for maximum
		"""
		# hide the right and top spines
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		# make ticks more visible
		ax.tick_params(which='major', length=10, width=3, labelsize=30)
		ax.tick_params(which='minor', length=4, width=2, labelsize=30)
		# set automatical locator for chosen axis
		if axis == 'x' or axis == 'both':
			ax.xaxis.set_minor_locator(MultipleLocator())
			if auto_nbins:
				ax.xaxis.set_major_locator(MaxNLocator(integer=True))
			else:
				ax.xaxis.set_major_locator(MaxNLocator(nbins=len(ax.get_xticks()), integer=True))

			if xshift or xmin or xmax:
				if xshift:
					xticks = (ax.get_xticks() + 1).astype(int)
				else:
					xticks = ax.get_xticks()
				xmax = np.inf if xmax is None else xmax
				xmin = -np.inf if xmin is None else xmin
				xticklabels = [int(label) if xmin <= label <= xmax else "" for label in xticks]
				ax.set_xticklabels(xticklabels)

		if axis == 'y' or axis == 'both':
			ax.yaxis.set_major_locator(MaxNLocator(nbins=len(ax.get_yticks()), integer=True))
			ax.yaxis.set_minor_locator(MultipleLocator(base=(ax.get_yticks()[1] - ax.get_yticks()[0]) / 2))

	@staticmethod
	def _shortname(name):
		"""
		Args:
			name (str):
		Returns:
		"""
		source, muscle, mode, speed, rate, pedal, stepsize = parse_filename(name)

		return f"{source}_{muscle}_{mode}_{speed}_{pedal}ped"

	@staticmethod
	def _ecdf(sample):
		# convert sample to a numpy array, if it isn't already
		sample = np.array(sample)
		# find the unique values and their corresponding counts
		quantiles, counts = np.unique(sample, return_counts=True)
		# take the cumulative sum of the counts and divide by the sample size to
		# get the cumulative probabilities between 0 and 1
		cumul_prob = np.cumsum(counts).astype(np.double) / sample.size

		return quantiles, cumul_prob

	@staticmethod
	def _find_extrema(array, condition):
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

	@staticmethod
	def _list3d(h, w):
		return [[[] for _ in range(w)] for _ in range(h)]

	def _get_peaks(self, sliced_datasets, dstep, borders, tails=False, debug=False):
		"""
		Finds all peaks times and amplitudes at each slice
		Args:
			sliced_datasets (np.ndarray):
			dstep (float): data step size
			borders (list): time borders for searching peaks
			tails (bool): move the peaks of first 3 ms to the previous slice
			debug (bool): debugging flag
		Returns:
			list: 3D list of peak times  [experiment_index][slice_index][peak times]
			list: 3D list of peak ampls  [experiment_index][slice_index][peak ampls]
			list: 3D list of peak slices [experiment_index][slice_index][peak slices indices]
		"""
		if type(sliced_datasets) is not np.ndarray:
			raise TypeError("Non valid type of data - use only np.ndarray")

		# form parameters for filtering peaks
		min_ampl = 0.3
		min_dist = int(0.15 / dstep)
		max_dist = int(4 / dstep)
		# interpritate shape of dataset
		tests_count, slices_count, slice_length = sliced_datasets.shape
		peak_per_slice_list = self._list3d(h=tests_count, w=slices_count)
		ampl_per_slice_list = self._list3d(h=tests_count, w=slices_count)
		peak_slice_num_list = self._list3d(h=tests_count, w=slices_count)
		# find all peaks times and amplitudes per slice
		if debug:
			plt.figure(figsize=(16, 9))

		for experiment_index, slices_data in enumerate(sliced_datasets):
			# combine slices into one myogram
			y = np.array(slices_data).ravel()
			# find all extrema
			e_maxima_indexes, e_maxima_values = self._find_extrema(y, np.greater)
			e_minima_indexes, e_minima_values = self._find_extrema(y, np.less)
			# start pairing extrema from maxima
			if e_minima_indexes[0] < e_maxima_indexes[0]:
				comb = list(zip(e_maxima_indexes, e_minima_indexes[1:]))
				combA = list(zip(e_maxima_values, e_minima_values[1:]))
			else:
				comb = list(zip(e_maxima_indexes, e_minima_indexes))
				combA = list(zip(e_maxima_values, e_minima_values))

			# # process each extrema pair
			if debug:
				xticks = np.arange(len(y)) * dstep
				plt.plot(xticks, y, color='k')
				plt.plot(e_maxima_indexes * dstep, e_maxima_values, '.', color='r')
				plt.plot(e_minima_indexes * dstep, e_minima_values, '.', color='b')
				per_dT = np.percentile(np.abs(np.diff(np.array(comb))) * dstep, q=[25, 50, 75])
				per_dA = np.percentile(np.abs(np.diff(np.array(combA))), q=[25, 50, 75])
			for max_index, min_index in comb:
				max_value = e_maxima_values[e_maxima_indexes == max_index][0]
				min_value = e_minima_values[e_minima_indexes == min_index][0]
				dT = abs(max_index - min_index)
				dA = abs(max_value - min_value)
				# check the difference between maxima and minima
				if (min_dist <= dT <= max_dist) and dA >= 0.028 or dA >= min_ampl:
					slice_index = int(max_index // slice_length)
					peak_time = max_index - slice_length * slice_index
					# change slice index for "tails" peaks
					if tails and peak_time * dstep <= 3 and slice_index > 0:
						slice_index -= 1
						peak_time = max_index - slice_length * slice_index

					if debug:
						plt.plot(max_index * dstep, y[max_index], '.', color='k')
						plt.text(max_index * dstep, y[max_index], f"({peak_time * dstep:.1f}, {slice_index})")
					if borders[0] <= peak_time * dstep < borders[1]:
						peak_per_slice_list[experiment_index][slice_index].append(peak_time)
						ampl_per_slice_list[experiment_index][slice_index].append(dA)
						peak_slice_num_list[experiment_index][slice_index].append(slice_index)

						if debug:
							plt.plot([max_index * dstep, min_index * dstep], [max_value, max_value], ls='--', color='k')
							plt.plot([min_index * dstep, min_index * dstep], [max_value, min_value], ls='--', color='k')
							plt.plot(max_index * dstep, max_value, '.', color='r', ms=15)
							plt.plot(min_index * dstep, min_value, '.', color='b', ms=15)
							plt.text(max_index * dstep, max_value + 0.05,
							         f"dT {(min_index - max_index) * dstep:.1f}\ndA {dA:.2f}", fontsize=10)
			if debug:
				plt.show()

		return peak_per_slice_list, ampl_per_slice_list, peak_slice_num_list

	def plot_peaks_bar_intervals(self, pack_peaks_per_interval, names, save_to):
		"""
		ToDo add info
		Args:
			pack_peaks_per_interval (list of np.ndarrays): grouped datasets by different sim/bio data
			names (list): filenames of grouped datasets
			save_to (str): path for saving a built picture
		"""
		raise NotImplemented
		yticks = []
		datasets_number = len(names)
		dist_coef = 2
		bar_height = 0.1
		coef = 0.3 * datasets_number
		colors = ["#275b78", "#f27c2e", "#f2aa2e", "#472650", "#a6261d", "#287a72", "#2ba7b9"] * 10
		ms_intervals = np.array([[0, 3], [7, 10], [10, 15], [15, 20], [20, 25]])
		# form the new name of the file
		mode = "_".join(names[0].split("_")[1:-1])
		merged_names = "_".join(name.split("_")[0] for name in names)
		new_filename = f"{merged_names}_{mode}_bar.pdf"

		fig, ax = plt.subplots(figsize=(15, 5))

		for pack_index, pack_data in enumerate(pack_peaks_per_interval):
			# set color based on filename
			if "bio" in names[pack_index]:
				marker = '✚'
			elif "neuron" in names[pack_index]:
				marker = '★'
			elif "gras" in names[pack_index]:
				marker = '▴'
			elif "nest" in names[pack_index]:
				marker = '⚫'
			else:
				raise Exception("Can not set marker for data!")
			# plot bar chart
			for interval_index, interval_data in enumerate(pack_data.T):
				left = 0
				# plot stacked bars (intervals are stacked, slices separated)
				y_pos = interval_index * coef + dist_coef * pack_index * bar_height
				if pack_index == 0:
					yticks.append(y_pos)
				for slice_index, slice_data in enumerate(interval_data):
					plt.barh(y=y_pos, width=slice_data, left=left, color=colors[slice_index], height=bar_height, alpha=0.9)
					left += slice_data
				plt.text(x=left + 0.1, y=y_pos - bar_height / 4, s=marker)

		yticks = np.array(yticks) + datasets_number * bar_height - bar_height
		yticklabels = [f"{interval[0]}-{interval[1]}" for interval in ms_intervals]
		plt.yticks(yticks, yticklabels)
		axis_article_style(ax, axis='x')
		plt.yticks(fontsize=30)
		plt.tight_layout()
		plt.savefig(f"{save_to}/{new_filename}", dpi=250, format="pdf")
		plt.close()
		log.info(f"saved to {save_to}/{new_filename}")

	@staticmethod
	def _example_bio_sample(folder, filename):
		"""
		Return y-data of best bio sample. File must exists
		Args:
			folder (str): current folder with hdf5 files and best sample
			filename (str): best sample filename
		Returns:
			np.ndarray: y-data of best sample
		"""
		raise NotImplemented
		best_samplse_filename = f"{folder}/best_samples/{filename.replace('.hdf5', '')}"
		print(best_samplse_filename)

		if not os.path.exists(best_samplse_filename):
			# raise Exception(f"Where is best sample for bio data?! I can't find it here '{folder}'")
			ideal_sample = np.array([[0] * 250 for _ in range(22)])
			return ideal_sample

		bio_ideal_y_data = []
		# collect extensor data
		with open(best_samplse_filename) as file:
			for d in file.readlines():
				bio_ideal_y_data.append(list(map(float, d.split())))
		# collect flexor_data
		with open(best_samplse_filename.replace('e_', 'f_')) as file:
			for d in file.readlines():
				bio_ideal_y_data.append(list(map(float, d.split())))
		# convert list to array for more simplicity using
		ideal_sample = np.array(bio_ideal_y_data)

		return ideal_sample

	@staticmethod
	def _example_sample(latencies_matrix, peaks_matrix, step_size):
		"""
		ToDo add info
		Args:
			latencies_matrix (np.ndarray):
			peaks_matrix (np.ndarray):
			step_size (float): data step size
		Returns:
			int: index of sample
		"""
		raise NotImplemented
		ideal_example_index = 0
		peaks_sum = np.sum(peaks_matrix, axis=1)
		index = np.arange(len(peaks_sum))
		merged = np.array(list(zip(index, peaks_sum)))
		# at the top located experimental runs with the greatest number of peaks
		sorted_by_sum = merged[merged[:, 1].argsort()][::-1]
		for index, value in sorted_by_sum:
			index = int(index)
			# check difference between latencies -- how far they are from each other
			diff = np.diff(latencies_matrix[index] * step_size, n=1)
			# acceptable border is -3 .. 3 ms
			if all(map(lambda x: -3 <= x <= 3, diff)):
				ideal_example_index = index
				break
		return ideal_example_index

	@staticmethod
	def _get_KS_2samp_pvalue(y1, y2):
		dvalue, _ = ks_2samp(y1, y2)
		en = np.sqrt(len(y1) * len(y2) / (len(y1) + len(y2)))
		den = dvalue * en
		pvalue = kstwobign.sf(den)
		return pvalue

	@staticmethod
	def _kde_test(x1, y1, x2, y2):
		r_fct_string = f"""  
		KDE_test <- function(){{
			library("ks")
			x1 <- c({str(list(x1))[1:-1]})
			y1 <- c({str(list(y1))[1:-1]})
			x2 <- c({str(list(x2))[1:-1]})
			y2 <- c({str(list(y2))[1:-1]})
			mat1 <- matrix(c(x1, y1), nrow=length(x1))
			mat2 <- matrix(c(x2, y2), nrow=length(x2))
	
			res_time <- kde.test(x1=x1, x2=x2)$pvalue
			res_ampl <- kde.test(x1=y1, x2=y2)$pvalue
			res_2d <- kde.test(x1=mat1, x2=mat2)$pvalue
	
	        return(c(res_time, res_ampl, res_2d))
		}}
		"""
		r_pkg = STAP(r_fct_string, "r_pkg")
		return np.asarray(r_pkg.KDE_test())

	@staticmethod
	def _get_AD_2samp_pvalue(a, b):
		reject_at = 0
		significance_levels = [25, 10, 5, 2.5, 1]
		statistic, critical_values, ad_pvalue = anderson_ksamp([a, b])
		for percent, level in zip(significance_levels, critical_values):
			if statistic > level:
				reject_at = percent
		return ad_pvalue, reject_at

	@staticmethod
	def _multi_R_KDE_test(times, ampls, borders, fs, save_to, dstep):
		r1 = fs[0]
		r2 = fs[1] # source, rat, mode, speed
		name1 = f"{r1[0]}_{r1[2]}_{r1[3]}"
		name2 = f"{r2[0]}_{r2[2]}_{r2[3]}"
		save_to = f"{save_to}/{name1} and {name2}/{r1[1]} {r2[1]}"

		if not os.path.exists(save_to):
			os.makedirs(save_to)

		dat = []
		steps_number = 0
		for time0, amp0 in zip(times[0], ampls[0]):
			for time1, amp1 in zip(times[1], ampls[1]):
				t0, t1 = np.array(list(flatten(time0))) * dstep, np.array(list(flatten(time1))) * dstep
				a0, a1 = np.array(list(flatten(amp0))), np.array(list(flatten(amp1)))

				if len(t0) < 5 or len(t1) < 5:
					continue

				kde_pval_t, kde_pval_a, kde_pval_2d = kde_test(t0, a0, t1, a1)
				dat.append([kde_pval_t, kde_pval_a, kde_pval_2d])

				gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
				fig = plt.figure(figsize=(8, 6))
				title_ax = plt.subplot(gs[0, 1])
				title_ax.text(0.5, 0.5, f"T: {kde_pval_t:.3f}\nA: {kde_pval_a:.3f}\n2D: {kde_pval_2d:.3f}",
				        horizontalalignment='center',
				        verticalalignment='center',
				        fontsize=18, transform=title_ax.transAxes)
				for spine in ['top', 'right', 'left', 'bottom']:
					title_ax.spines[spine].set_visible(False)
				title_ax.set_xticks([])
				title_ax.set_yticks([])

				kde_ax = plt.subplot(gs[1, 0])
				kde_ax.spines['top'].set_visible(False)
				kde_ax.spines['right'].set_visible(False)

				# 2D joint plot
				z = []
				label_pathes = []
				z_prev = np.zeros(1)
				#
				for x, y, name, color in zip([t0, t1], [a0, a1], [name1, name2], ['#A6261D', '#472650']):
					z_prev = contour_plot(x=x, y=y, color=color, ax=kde_ax, z_prev=z_prev, borders=borders, levels_num=15)
					z.append(z_prev)
					t, r = joint_plot(x, y, kde_ax, gs, **{"color": color}, borders=borders, with_boxplot=False)
					label_pathes.append(mpatches.Patch(color=color, label=f"{name}"))

					t.set_xticklabels([])
					r.set_yticklabels([])

					t.set_xlim(borders[0], borders[1])
					r.set_ylim(borders[2], borders[3])

				kde_ax.legend(handles=label_pathes, fontsize=17)
				kde_ax.set_xlim(borders[0], borders[1])
				kde_ax.set_ylim(borders[2], borders[3])

				plt.tight_layout()
				plt.savefig(f"{save_to}/{steps_number}.png", format="png")
				plt.close(fig)

				steps_number += 1

		dat = np.array(dat)
		a = f"T: median p-value={np.median(dat[:, 0]):.3f}. Passed steps: ({len(np.where(dat[:, 0] >= 0.05)[0]) / len(dat) * 100:.1f}%)"
		b = f"A: median p-value={np.median(dat[:, 1]):.3f}. Passed steps: ({len(np.where(dat[:, 1] >= 0.05)[0]) / len(dat) * 100:.1f}%)"
		c = f"2D: median p-value={np.median(dat[:, 2]):.3f}. Passed steps: ({len(np.where(dat[:, 2] >= 0.05)[0]) / len(dat) * 100:.1f}%)"
		log.info(a)
		log.info(b)
		log.info(c)

		fig = plt.figure(figsize=(5, 5))
		plt.suptitle(f"{a}\n{b}\n{c}", fontsize=10)
		plt.boxplot(dat[:, 0], positions=[0], widths=0.8)
		plt.boxplot(dat[:, 1], positions=[1], widths=0.8)
		plt.boxplot(dat[:, 2], positions=[2], widths=0.8)
		plt.xticks([0, 1, 2], ["times", "ampls", "kde2d"])
		plt.savefig(f"{save_to}/pval_boxplot.png", format="png")
		plt.close(fig)

		with open(f"{save_to}/stat", 'w') as file:
			for t, a, d2 in dat:
				file.write(f"{t:.5f}\t{a:.5f}\t{d2:.5f}\n")

	@staticmethod
	def _contour_plot(x, y, color, ax, z_prev, borders, levels_num):
		"""

		Args:
			x:
			y:
			color:
			ax:
			z_prev:
			borders:
		Returns:
			np.ndarray:
		"""
		xmin, xmax, ymin, ymax = borders
		# form a mesh grid
		xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		# re-present grid in 1D and pair them as (x1, y1 ...)
		positions = np.vstack([xx.ravel(), yy.ravel()])
		values = np.vstack([x, y])
		# use a Gaussian KDE
		a = st.gaussian_kde(values)(positions).T
		# re-present grid back to 2D
		z = np.reshape(a, xx.shape)
		# find an index of the maximal element
		m = np.amax(z)
		# form a step and levels
		step = (np.amax(a) - np.amin(a)) / levels_num
		levels = np.arange(0, m, step) + step
		# convert HEX to HSL
		h_norm, s_norm, l_norm = Color(color).hsl
		# make an gradient based on number of levels
		light_gradient = np.linspace(l_norm, 0.95, len(levels))[::-1]
		# generate colors for contours level from HSL (normalized) to RGB (normalized)
		colors = [Color(hsl=(h_norm, s_norm, l_level)).rgb for l_level in light_gradient]
		# plot filled contour
		if len(z_prev) == 1:
			zorder = 0
		else:
			gt = z.ravel() > z_prev.ravel()
			fullsize = len(gt)
			gt = gt[gt == True]
			truesize = len(gt)
			zorder = -5 if truesize / fullsize * 100 > 50 else 5

		ax.contour(xx, yy, z, levels=levels, linewidths=1, colors=color)
		ax.contourf(xx, yy, z, levels=levels, colors=colors, alpha=0.7, zorder=zorder)
		# ax.scatter(x, y, s=0.1, color=color)
		return z

	@staticmethod
	def _get_color(filename, clrs):
		if "bio" in filename:
			color = next(clrs)
		elif "gras" in filename:
			color = '#287a72'
		elif "neuron" in filename:
			color = '#F2AA2E'
		elif "nest" in filename:
			color = '#472650'
		else:
			raise Exception("Can't set color for data")
		return color
