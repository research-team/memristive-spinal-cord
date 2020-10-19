import os
import pickle
import logging
import numpy as np
import pylab as plt
import scipy.stats as st
import plotly.offline as py
import plotly.graph_objects as go
import matplotlib.patches as mpatches
from scipy import interpolate
import math

from colour import Color
from itertools import chain
from shapely import affinity
from matplotlib import gridspec
from collections import defaultdict
from matplotlib.patches import Polygon
from scipy.signal import argrelextrema
from matplotlib.patches import Ellipse
from rpy2.robjects.packages import STAP
from shapely.geometry.point import Point
from scipy.stats import ks_2samp, kstwobign
from rpy2.robjects.numpy2ri import numpy2rpy
from matplotlib.ticker import MaxNLocator, MultipleLocator
from analysis.functions import read_hdf5, trim_data, calibrate_data, get_boxplots, parse_filename, subsampling

flatten = chain.from_iterable

class Metadata:
	def __init__(self, pickle_folder=None, filename=None, sdict=None):
		if sdict:
			self.pdata = sdict
		else:
			if pickle_folder and filename:
				self.pdata = self.unpickle_data(pickle_folder, filename)
			else:
				raise Exception(f"Check the filename '{filename}' and folder '{pickle_folder}'")

	def unpickle_data(self, pickle_folder, filename):
		filename = filename if ".pickle" in filename else f"{filename}.pickle"
		with open(f"{pickle_folder}/{filename.replace('_E_', '_')}", 'rb') as handle:
			return pickle.load(handle)

	def get_rats_id(self, muscle='E'):
		return self.pdata['rats_data'][muscle].keys()

	def get_myograms(self, rat, muscle='E'):
		return self.pdata['rats_data'][muscle][rat]['data']

	def get_peak_times(self, rat, muscle='E', flat=False, unslice=False):
		data = self.pdata['rats_data'][muscle][rat]['times']
		if flat:
			return np.array(list(flatten(flatten(data))))
		if unslice:
			return [list(flatten(d)) for d in data]
		return data

	def get_peak_durations(self, rat, muscle='E', flat=False, unslice=False):
		data = self.pdata['rats_data'][muscle][rat]['durations']
		if flat:
			return np.array(list(flatten(flatten(data))))
		if unslice:
			return [list(flatten(d)) for d in data]
		return data

	def get_peak_ampls(self, rat, muscle='E', flat=False, unslice=False):
		data = self.pdata['rats_data'][muscle][rat]['ampls']
		if flat:
			return np.array(list(flatten(flatten(data))))
		if unslice:
			return [list(flatten(d)) for d in data]
		return data

	def get_peak_slices(self, rat, muscle='E', flat=False, unslice=False):
		data = self.pdata['rats_data'][muscle][rat]['slices']
		if flat:
			return np.array(list(flatten(flatten(data))))
		if unslice:
			return [list(flatten(d)) for d in data]
		return data

	def get_fMEP_count(self, rat, muscle='E'):
		return len(self.pdata['rats_data'][muscle][rat]['data'])

	def get_peak_counts(self, rat, border, muscle='E'):
		times = self.get_peak_times(rat, muscle, flat=True) * self.pdata['dstep_to']
		fMEPs = self.get_fMEP_count(rat, muscle)

		if border == 'poly_tail':
			# [0 - 3] and [8 - 25]
			mask = (times <= 3) | (8 <= times)
		else:
			# [T1, T2]
			mask = (border[0] <= times) & (times <= border[1])

		return len(times[mask]) / fMEPs

	def get_peak_median_height(self, rat, border, muscle='E'):
		ampls = self.get_peak_ampls(rat, muscle, flat=True)
		times = self.get_peak_times(rat, muscle, flat=True) * self.pdata['dstep_to']
		if border == 'poly_tail':
			# [0 - 3] and [8 - 25]
			mask = (times <= 3) | (8 <= times)
		else:
			# [T1, T2]
			mask = (border[0] <= times) & (times <= border[1])

		return np.median(ampls[mask])

	def get_latency_volume(self, rat, muscle='E'):
		return self.pdata['rats_data'][muscle][rat]['latency_volume']

	@property
	def shortname(self):
		return self.pdata['shortname']

	@property
	def rate(self):
		return self.pdata['rate']

	@property
	def dstep_to(self):
		return self.pdata['dstep_to']

	@property
	def dstep_from(self):
		return self.pdata['dstep_from']

	@property
	def slice_in_ms(self):
		return self.pdata['slice_in_ms']

	@property
	def slice_count(self):
		return self.pdata['slices_count']

	@property
	def speed(self):
		return self.pdata['speed']


class Analyzer:
	def __init__(self, pickle_folder, debug=False):
		self.pickle_folder = pickle_folder
		self.plots_folder = f"{pickle_folder}/plots"

		self.debug = debug

		if not os.path.exists(self.pickle_folder):
			os.makedirs(self.pickle_folder)
		if not os.path.exists(self.plots_folder):
			os.makedirs(self.plots_folder)

		logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
		self.log = logging.getLogger()

	@staticmethod
	def disable_log():
		logging.disable(logging.CRITICAL)

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

	def KDE_plot(self, xy1, xy2, names, rats, border):
		plt.close()
		gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
		plt.figure(figsize=(8, 6))
		title_ax = plt.subplot(gs[0, 1])

		for spine in ['top', 'right', 'left', 'bottom']:
			title_ax.spines[spine].set_visible(False)
		title_ax.set_xticks([])
		title_ax.set_yticks([])

		kde_ax = plt.subplot(gs[1, 0])
		kde_ax.spines['top'].set_visible(False)
		kde_ax.spines['right'].set_visible(False)
		label_pathes = []
		#
		for (x, y), name, color in zip([xy1, xy2],
		                               [f"{names[0]} {rats[0]}", f"{names[1]} {rats[1]}"],
		                               ['#A6261D', '#472650']):
			self._contour_plot(x=x, y=y, color=color, ax=kde_ax, z_prev=[0],
			                   borders=border, levels_num=15)
			t, r = self.joint_plot(x, y, kde_ax, gs, **{"color": color}, borders=border, with_boxplot=False)
			label_pathes.append(mpatches.Patch(color=color, label=f"{name}"))
			# kde_ax.plot(x, y, '.', c=color)
			t.set_xticklabels([])
			r.set_yticklabels([])

			t.set_xlim(border[0], border[1])
			r.set_ylim(border[2], border[3])

		kde_ax.legend(handles=label_pathes, fontsize=17)
		kde_ax.set_xlim(border[0], border[1])
		kde_ax.set_ylim(border[2], border[3])
		plt.tight_layout()

	def outside_compare(self, comb, border, axis, muscletype='E', per_step=False, plot=False, show=False):
		if len(comb) != 2:
			raise Exception("Only pairing analysis")
		#
		metadata1 = Metadata(self.pickle_folder, comb[0][0])
		metadata2 = Metadata(self.pickle_folder, comb[1][0])

		# do not change an order!
		axis_names = ('time', 'ampl', 'slice')  # do not change an order!
		axis_borders = ([8, 28] if border == 'poly_tail' else border,
		                (0, 1.5),
		                (-1, max(metadata1.slice_count, metadata2.slice_count) + 1))
		#
		ax1_index = axis_names.index(axis[0])
		ax2_index = axis_names.index(axis[1])

		meta_names = (metadata1.shortname, metadata2.shortname)
		#
		for rat1 in comb[0][1]:
			for rat2 in comb[1][1]:
				# check if pval file is exist to save a calulcation time
				if per_step:
					"""
					Here p-value will be calculated for each step in R KDE test.
					Convertation lists with different lengths to matrix is only way to pass the matrix to R.
					Because of that the emptines filled by -9999 (trash) values, whtich will be filtered in R.
					np.where is used instead of classic slice because of saving true shape of the matrix
					"""
					dataXY = []

					for r, m in zip((rat1, rat2), (metadata1, metadata2)):
						t = m.get_peak_times(rat=r, unslice=True, muscle='E')
						a = m.get_peak_ampls(rat=r, unslice=True, muscle='E')
						s = m.get_peak_slices(rat=r, unslice=True, muscle='E')
						maxlen = max(map(len, t))
						t = np.array([d + [-9999] * (maxlen - len(d)) for d in t]) * m.dstep_to
						a = np.array([d + [-9999] * (maxlen - len(d)) for d in a])
						s = np.array([d + [-9999] * (maxlen - len(d)) for d in s])
						shape = t.shape

						if border == 'poly_tail':
							t[t <= 3] += 25 # all peaks at [0, 3] becomes [25, 28]
							mask = t >= 8
						else:
							mask = (border[0] <= t) & (t <= border[1])

						# pick the data which corresponds to the axis name with saving shape
						data = (np.where(mask, t, -9999).reshape(shape),
						        np.where(mask, a, -9999).reshape(shape),
						        np.where(mask, s, -9999).reshape(shape))
						X, Y = data[ax1_index], data[ax2_index]

						# number of point must be more than 16!
						not_good_rows = [i for i, d in enumerate(X) if len(d[d >= 0]) < 10]

						X = np.delete(X, not_good_rows, axis=0)
						Y = np.delete(Y, not_good_rows, axis=0)

						if len(X) == 0:
							print('EMPTY')
							break
						dataXY.append((X, Y))
					else:
						# !
						# todo about .T
						pval_t, pval_a, pval_2d = self._multi_R_KDE_test(*dataXY[0], *dataXY[1]).T
						print(f"{meta_names[0]} rat {rat1} vs {meta_names[1]} rat {rat2} "
						      f"'{axis[0]} by step': {np.median(pval_t):.3f}; "
						      f"'{axis[1]} by step': {np.median(pval_a):.3f}; "
						      f"2D by step: {np.median(pval_2d):.3f}")
						# plot data if necessary
						if plot:
							kde_border = (*axis_borders[ax1_index], *axis_borders[ax2_index])
							for xstep1, ystep1 in zip(*dataXY[0]):
								for xstep2, ystep2 in zip(*dataXY[1]):
									xy1 = xstep1[xstep1 >= 0], ystep1[ystep1 >= 0]
									xy2 = xstep2[xstep2 >= 0], ystep2[ystep2 >= 0]
									self.KDE_plot(xy1, xy2, meta_names, (rat1, rat2), kde_border)
									filename = f"{meta_names[0]}_rat{rat1}_{meta_names[1]}_rat{rat2}_merged"
									plt.savefig(f'{self.plots_folder}/{filename}.pdf', format='pdf')
									if show:
										plt.show()
									plt.close()
				else:
					"""
					The simpliest way to compare data -- 1D representation.
					That way all steps merged into one dataset.
					P-value will critical differ from median p-value of step-by-step analysis
					"""
					dataXY = []
					for r, m in zip((rat1, rat2), (metadata1, metadata2)):
						t = m.get_peak_times(rat=r, flat=True, muscle=muscletype) * m.dstep_to
						a = m.get_peak_ampls(rat=r, flat=True, muscle=muscletype)
						s = m.get_peak_slices(rat=r, flat=True, muscle=muscletype)
						if border == 'poly_tail':
							t[t <= 3] += 25 # all peaks at [0, 3] becomes [25, 28]
							mask = 8 <= t
						else:
							mask = (border[0] <= t) & (t <= border[1])
						# pick the data which corresponds to the axis name
						d = (t[mask], a[mask], s[mask])
						X, Y = d[ax1_index], d[ax2_index]
						dataXY.append((X, Y))
					# plt.close()
					# print(m.shortname, r)
					# for si in range(0, max(s) + 1):
					# 	plt.boxplot(a[s == si], positions=[si], whis=[5, 95])
					# plt.ylim(0, 1.7)
					# plt.show()
					# calc the p-value by KDE test
					pval_t, pval_a, pval_2d = self._multi_R_KDE_test(*dataXY[0], *dataXY[1])
					print(f"{meta_names[0]} rat {rat1} vs {meta_names[1]} rat {rat2} - {muscletype} muscle"
					      f"'{axis[0]} merged': {pval_t}; "
					      f"'{axis[1]} merged': {pval_a}; "
					      f"2D merged: {pval_2d}")
					# plot data if necessary
					if plot:
						kde_border = (*axis_borders[ax1_index], *axis_borders[ax2_index])
						self.KDE_plot(dataXY[0], dataXY[1], meta_names, (rat1, rat2), kde_border)
						filename = f"{meta_names[0]}_rat{rat1}_{meta_names[1]}_rat{rat2}_merged"
						plt.savefig(f'{self.plots_folder}/{filename}.pdf', format='pdf')
						if show:
							plt.show()
						plt.close()

	def plot_cumulative(self, cmltv, border, order=None, pval_slices_peak=False):
		"""
		"""
		data = []
		wspace = 6
		pos_dict = []
		clr_height = '#472650'
		clr_count = '#a6261d'
		clr_volume = '#287a72'
		global_significancies = defaultdict(list)
		names = np.array(cmltv)[:, :2]
		uniq_names = list(np.unique(names))

		if order:
			if sorted(uniq_names) != sorted(order):
				raise Exception("Order is not contain all uniq names of 'cmltv' list")
			uniq_names = order

		def add_signific(source, target, value):
			#
			if source > target:
				source, target = target, source
			#
			top = max(pos_dict[source:target + 1]) + tickrisk
			global_significancies[target - source].append([source, target, top, value])
			global_significancies[target - source] = sorted(global_significancies[target - source], key=lambda d: d[2])

		def apply_signific():
			last_max = 0
			#
			for key in sorted(global_significancies.keys()):
				for d in global_significancies[key]:
					if d[2] > last_max:
						last_max = d[2]
					else:
						while d[2] <= last_max:
							d[2] += 3 * tickrisk
						last_max = d[2]
			#
			for meta in sum(global_significancies.values(), []):
				xl, xr, yt, textval = meta
				yt += 3 * tickrisk
				ax1.plot(np.array((xl + 0.05, xl + 0.05, xr - 0.05, xr - 0.05)) * wspace,
				         (yt - 1.2 * tickrisk, yt, yt, yt - 1.2 * tickrisk), lw=2, c='k')
				ax1.text(np.mean([xr, xl]) * wspace, yt + tickrisk, textval, c='k', ha='center', fontsize=15)

			return last_max
		#
		for uniq_key in uniq_names:
			filename = [f for f in os.listdir(self.pickle_folder) if uniq_key in f and f.endswith(".pickle")][0]
			pdata = Metadata(self.pickle_folder, filename)
			rats = pdata.get_rats_id(muscle='E')
			counts = [pdata.get_peak_counts(rat, border=border) for rat in rats]
			heights = [pdata.get_peak_median_height(rat, border=border) for rat in rats]
			volumes = [pdata.get_latency_volume(rat) for rat in rats]
			for rat_id, count, height, volume in zip(rats, counts, heights, volumes):
				data.append([uniq_key, rat_id, count, height, volume])

		grouped_height = {k: [] for k in uniq_names}
		grouped_count = {k: [] for k in uniq_names}
		grouped_volume = {k: [] for k in uniq_names}

		for d in data:
			grouped_count[d[0]].append(d[2])
			grouped_height[d[0]].append(d[3])
			grouped_volume[d[0]].append(d[4])

		all_vals_ax1 = [d[2] for d in data]
		all_vals_ax2 = [d[3] for d in data]
		all_vals_ax3 = [d[4] for d in data]

		ax1_min, ax1_max = min(all_vals_ax1), max(all_vals_ax1)
		ax2_min, ax2_max = min(all_vals_ax2), max(all_vals_ax2)
		ax3_min, ax3_max = min(all_vals_ax3), max(all_vals_ax3)

		tickrisk = (ax1_max - ax1_min) * 0.02

		fig, ax1 = plt.subplots(figsize=(16, 9))
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		ax3 = ax1.twinx()
		ax3.spines["right"].set_position(("axes", 1.1))

		i = 0
		averagesX = []
		averagesY = []
		for name, c, h, v in zip(uniq_names, grouped_count.values(), grouped_height.values(), grouped_volume.values()):
			i1, i2, i3 = i - 1, i, i + 1
			# make the same data ratio
			h = [(ax1_max - ax1_min) * (h_elem - ax2_min) / (ax2_max - ax2_min) + ax1_min for h_elem in h]
			v = [(ax1_max - ax1_min) * (v_elem - ax3_min) / (ax3_max - ax3_min) + ax1_min for v_elem in v]
			# for trendlines
			averagesX.append((i1 - 0.3, i2 - 0.3, i3 - 0.3))
			averagesY.append((np.mean(c), np.mean(h), np.mean(v)))

			for ax, index, dat, color in zip([ax1, ax2, ax3], [i1, i2, i3],
			                                 [c, h, v], [clr_count, clr_height, clr_volume]):
				ax.plot([index - 0.4, index - 0.2], [np.mean(dat)] * 2, color=color, lw=4)
				ax.plot([index - 0.3] * 2, [max(dat), min(dat)], color=color, lw=1.5)
				ax.plot([index] * len(dat), dat, '.', ms=15, color=color)

			pos_dict.append(max(h + c + v))
			i += wspace

		for xx, yy, clr in zip(np.array(averagesX).T, np.array(averagesY).T, [clr_count, clr_height, clr_volume]):
			ax.plot(xx, yy, c=clr, alpha=0.4)

		if len(cmltv[0]) == 3:
			for p1, p2, pval in cmltv:
				logging.info(f"Pair {p1} and {p2}")
				source = uniq_names.index(p1)
				target = uniq_names.index(p2)
				add_signific(source, target, value=f"{pval:.2f}")
		else:
			for p1, p2 in cmltv:
				logging.info(f"Pair {p1} and {p2}")
				source = uniq_names.index(p1)
				target = uniq_names.index(p2)

				filename1 = [f for f in os.listdir(self.pickle_folder) if p1 in f and f.endswith(".pickle")][0]
				filename2 = [f for f in os.listdir(self.pickle_folder) if p2 in f and f.endswith(".pickle")][0]
				pdata1 = self.get_pickle_data(filename1)
				pdata2 = self.get_pickle_data(filename2)

				dstep_to1 = pdata1['dstep_to']
				dstep_to2 = pdata2['dstep_to']

				pvalues = []
				for rat1 in self.get_rats_id(pdata1):
					for rat2 in self.get_rats_id(pdata2):
						# check if pval file is exist to save a calulcation time
						pval_file = f"{self.pickle_folder}/pval_ampl_{p1}_{rat1}+{p2}_{rat2}"

						if os.path.exists(pval_file) and os.path.getsize(pval_file) > 0:
							with open(f"{self.pickle_folder}/pval_ampl_{p1}_{rat1}+{p2}_{rat2}") as file:
								pval_x, pval_y, pval_2d = [], [], []
								for line in file.readlines():
									pval_x, pval_y, pval_2d = list(map(float, line.split("\t")))
						else:
							x1 = [(np.array(d) * dstep_to1).tolist() for d in self.get_peak_times(pdata1, rat=rat1, unslice=True)]
							x2 = [(np.array(d) * dstep_to2).tolist() for d in self.get_peak_times(pdata2, rat=rat2, unslice=True)]

							if pval_slices_peak:
								y1 = self.get_peak_slices(pdata1, rat=rat1, unslice=True)
								y2 = self.get_peak_slices(pdata2, rat=rat2, unslice=True)
							else:
								y1 = self.get_peak_ampls(pdata1, rat=rat1, unslice=True)
								y2 = self.get_peak_ampls(pdata2, rat=rat2, unslice=True)

							pval_x, pval_y, pval_2d = self._multi_R_KDE_test(x1, y1, x2, y2)
							# save pvalues
							with open(pval_file, 'w') as file:
								for px, py, p2d in zip(pval_x, pval_y, pval_2d):
									file.write(f"{px:.5f}\t{py:.5f}\t{p2d:.5f}\n")
							#
							ind = 0
							for aa, bb in zip(x1, y1):
								for cc, dd in zip(x2, y2):
									kde_pval_t = pval_x[ind]
									kde_pval_a = pval_y[ind]
									kde_pval_2d = pval_2d[ind]
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
									border = [8, 28, 0, 1.4]
									for x, y, name, color in zip([np.array(aa) * dstep_to1, np.array(cc) * dstep_to2],
									                             [bb, dd], [f"{p1} {rat1}", f"{p2} {rat2}"], ['#A6261D', '#472650']):
										z_prev = self._contour_plot(x=x, y=y, color=color, ax=kde_ax, z_prev=z_prev, borders=border,
										                            levels_num=15)
										z.append(z_prev)
										t, r = self.joint_plot(x, y, kde_ax, gs, **{"color": color}, borders=border, with_boxplot=False)
										label_pathes.append(mpatches.Patch(color=color, label=f"{name}"))

										t.set_xticklabels([])
										r.set_yticklabels([])

										t.set_xlim(border[0], border[1])
										r.set_ylim(border[2], border[3])
										kde_ax.plot(x, y, '.', color=color)

									kde_ax.legend(handles=label_pathes, fontsize=17)
									kde_ax.set_xlim(border[0], border[1])
									kde_ax.set_ylim(border[2], border[3])

									plt.tight_layout()
									plt.show()
									plt.close(fig)
									ind += 1
						# end if
						pvalues.append((rat1, rat2, np.median(pval_2d)))
						logging.info(f"{p1}_{rat1} vs {p2}_{rat2} {np.median(pval_x)} {np.median(pval_y)} {np.median(pval_2d)}")
				# end rats for loop
				pvals = [p[2] for p in pvalues]
				add_signific(source, target, value=f"{np.median(pvals):.2f}")
			# end p1, p2 for loop
		# end if
		max_val = apply_signific()

		# make more readable
		ax1.set_ylim(10, max_val + 5 * tickrisk)
		ax2.set_ylim(10, max_val + 5 * tickrisk)
		ax3.set_ylim(10, max_val + 5 * tickrisk)

		# back to original tick labels
		new_ax2_ticklabels = [(ax2_max - ax2_min) * (tick - ax1_min) / (ax1_max - ax1_min) + ax2_min for tick in
		                      ax2.yaxis.get_major_locator()()]
		new_ax3_ticklabels = [(ax3_max - ax3_min) * (tick - ax1_min) / (ax1_max - ax1_min) + ax3_min for tick in
		                      ax3.yaxis.get_major_locator()()]
		ax2.set_yticklabels(np.round(new_ax2_ticklabels, 2))
		ax3.set_yticklabels(np.round(new_ax3_ticklabels, 2))

		for ax, color, label in zip([ax1, ax2, ax3],
		                            [clr_count, clr_height, clr_volume],
		                            ['Number of peaks per fMEP', 'Median peak height', 'Latency volume']):
			ax.set_ylabel(label, color=color, fontsize=23)
			ax.tick_params(axis='y', labelcolor=color, labelsize=20)
			ax.tick_params(axis="x", labelsize=20)

			ax.spines['top'].set_visible(False)
		xticks = ["\n".join(xtick.split("_")[1:]) for xtick in uniq_names]
		plt.xticks(np.arange(len(uniq_names)) * wspace, xticks, fontsize=15)
		plt.tight_layout()
		plt.savefig(f"{self.plots_folder}/cumulative.pdf", format='pdf')
		plt.show()

	def _lw_prepare_data(self, folder, muscle, metadata, fill_zeros, filter_val, hz_analysis=None):
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
		if hz_analysis:
			slise_in_ms = 50

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
				prepared_data = np.array(calibrate_data(dataset, source))
				# print(len(prepared_data))
			if hz_analysis:
				if muscle == "E":
					K = []
					for i in range(len(prepared_data)):
						stepdata = prepared_data[i]
						ees_int = int(1000 / metadata['rate'])
						print(ees_int)
						slice_len = ((50 // ees_int) * ees_int)
						# print(slice_len)
						slice_frame = slice_len / dstep_to
						# print(slice_frame)
						if slice_frame == 0:
							slice_frame = 50 / dstep_to
						slices_begin_indexes = range(0, len(stepdata) + 1, int(slice_frame))
						# for beg in slices_begin_indexes:
						# 	print(beg)
						splitted_per_slice = [stepdata[beg:(beg + int(50 / dstep_to))] for beg in slices_begin_indexes]
						splitted_per_slice = splitted_per_slice[:6]
						# remove tails
						# print(list(map(len, splitted_per_slice)))
						K.append(np.array(splitted_per_slice))
				else:
					K = []
					for i in range(3):
						stepdata = prepared_data[i]
						ees_int = int(1000 / metadata['rate'])
						slice_frame = ((50 // ees_int) * ees_int) / dstep_to
						if slice_frame == 0:
							slice_frame = 50 / dstep_to
						slices_begin_indexes = range(0, len(stepdata) + 1, int(slice_frame))
						splitted_per_slice = [stepdata[beg:beg + int(50 / dstep_to)] for beg in slices_begin_indexes]
						splitted_per_slice = splitted_per_slice[:2]
						# remove tails
						# print(list(map(len, splitted_per_slice)))
						K.append(np.array(splitted_per_slice))
				sliced_data = np.array(K)
			else:
				if muscle == "E":
					sliced_data = [np.array_split(beg, e_slices_number) for beg in prepared_data]
				else:
					sliced_data = [np.array_split(beg, f_slices_number) for beg in prepared_data]
				sliced_data = np.array(sliced_data)

			# print(sliced_data.shape)

			if len(sliced_data) == 0:
				metadata['rats_data'][muscle][rat_id] = dict(data=None,
				                                             times=None,
				                                             ampls=None,
				                                             slices=None,
				                                             latency_volume=None)
				continue
			#
			sliced_time, sliced_ampls, sliced_index = self._get_peaks(sliced_data, dstep_to,
			                                                          [0, slise_in_ms], filter_val, debug=self.debug)

			metadata['rats_data'][muscle][rat_id] = dict(data=sliced_data,
			                                             times=sliced_time,
			                                             ampls=sliced_ampls,
			                                             slices=sliced_index)

			# do not calculate volume for FLEXOR (is redundant)
			if muscle == 'E':
				latency_volume = None#self.plot_density_3D(source=metadata, rats=rat_id, factor=15, only_volume=True)[0]
			else:
				latency_volume = None

			metadata['rats_data'][muscle][rat_id]['latency_volume'] = latency_volume

	#
	def prepare_data(self, folder, dstep_to=None, fill_zeros=True, filter_val=0.05, hz_analysis=False):
		"""

		Args:
			folder:
			dstep_to:
			fill_zeros:
			filter_val:
		"""
		# check each .hdf5 file in the folder
		for filename in [f for f in os.listdir(folder) if f.endswith('.hdf5') and '_E_' in f]:
			source, muscle, mode, speed, rate, pedal, dstep = parse_filename(filename)
			shortname = f"{source}_{mode}_{speed}_{rate}hz_{pedal}ped"
			#
			if dstep_to is None:
				dstep_to = dstep
			# prepare the metadata dict
			metadata = {
				'filename': filename,
				'source': source,
				'muscle': muscle,
				'mode': mode,
				'speed': speed,
				'rate': rate,
				'slice_in_ms': 1 / rate * 1000,
				'pedal': pedal,
				'dstep_from': dstep,
				'dstep_to': dstep_to,
				'shortname': shortname,
				'rats_data': {
					'E': {},
					'F': {}
				}
			}
			# fill the metadata for each muscle (number of peaks, median height and etc)
			self._lw_prepare_data(folder, 'E', metadata, fill_zeros, filter_val, hz_analysis)
			self._lw_prepare_data(folder, 'F', metadata, fill_zeros, filter_val, hz_analysis)
			#
			any_rat = list(metadata['rats_data'][muscle].keys())[0]
			metadata['slices_count'] = len(metadata['rats_data']['E'][any_rat]['data'][0])
			# save metadata as pickle object (dict)
			pickle_save = f"{self.pickle_folder}/{os.path.splitext(filename.replace('_E_', '_'))[0]}.pickle"
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

	def print_metainfo(self, source, rats):
		if type(source) is not Metadata:
			metadata = Metadata(self.pickle_folder, source)
		else:
			metadata = source

		if rats is None or rats is all:
			rats = metadata.get_rats_id()
		if type(rats) is int:
			rats = [rats]

		for muscle in ['E', 'F']:
			print("Filename | rat id | fMEPs | number of peaks per fMEP | median peak height | latency volume")
			for rat_id in rats:
				c = metadata.get_peak_counts(rat_id, border=[0, 25], muscle=muscle)
				h = metadata.get_peak_median_height(rat_id, border='poly_tail', muscle=muscle)
				f = metadata.get_fMEP_count(rat_id, muscle=muscle)
				v = metadata.get_latency_volume(rat_id, muscle=muscle)
				print(f"{metadata.shortname} _ {muscle} | {rat_id} | {f} | {c} | {h} | {v}")

				x = metadata.get_peak_times(rat_id, muscle=muscle, flat=True) * metadata.dstep_to
				# x = metadata.get_peak_ampls(rat_id, muscle='F', flat=True) * 100
				X = np.linspace(0, 25, 100)
				dx = st.gaussian_kde(x)
				dx.set_bandwidth(bw_method=0.175)
				dx = dx(X)
				# modes = np.array(self._find_extrema(dx, np.greater)[0]) * 25 / 100
				# distr = {1: 'uni', 2: 'bi'}
				# print(f"{metadata.shortname} #{rat_id} ({distr.get(len(modes), 'multi')}modal): {modes} ms")
				plt.plot(np.arange(len(dx)) * 0.25, dx, label=metadata.shortname)
			print("- " * 10)

	def plot_fMEP_boxplots(self, source, borders, rats=None, show=False, slice_ms=None):
		if type(source) is not Metadata:
			metadata = Metadata(self.pickle_folder, source)
		else:
			metadata = source

		if rats is None or rats is all:
			rats = metadata.get_rats_id()
		if type(rats) is int:
			rats = [rats]

		speed = metadata.speed
		dstep_to = metadata.dstep_to
		shortname = metadata.shortname
		if slice_ms is None:
			slice_in_ms = metadata.slice_in_ms
		else:
			slice_in_ms = slice_ms

		# plot per each rat
		for rat_id in rats:
			rat_myograms = metadata.get_myograms(rat_id, muscle='E')
			rat_peak_times = metadata.get_peak_times(rat_id, muscle='E')

			total_rat_steps = rat_myograms.shape[0]
			total_slices = rat_myograms.shape[1]
			total_datasteps = rat_myograms.shape[2]

			plt.close()
			if speed == "6":
				fig, ax = plt.subplots(figsize=(20, 20))
			elif speed == "13.5":
				fig, ax = plt.subplots(figsize=(16, 12))
			else:
				fig, ax = plt.subplots(figsize=(16, 8))

			colors = iter(["#275b78", "#f2aa2e", "#a6261d", "#472650"] * total_rat_steps)

			xticks = np.arange(total_datasteps) * dstep_to
			# plot sliced myogram data
			for myogram_fMEP in rat_myograms:
				color = next(colors)
				for slice_index, slice_data in enumerate(myogram_fMEP):
					plt.plot(xticks, np.array(slice_data) + slice_index, alpha=0.5, color=color, zorder=1)

			# for each border (if it is a list of lists) find peaks inside and form boxplots
			if type(borders[0]) is not list:
				borders = [borders]
			for border in borders:
				# meta info about percent of existing at least on peak in the border
				passed = 0
				alles = total_rat_steps * total_slices
				# prepare lists for boxplots forming
				sliced_x = [[] for _ in range(total_slices)]
				# find peaks and plot them
				for myogram_fMEP, time_per_step in zip(rat_myograms, rat_peak_times):
					for slice_index, (slice_data, peak_time_per_slice) in enumerate(zip(myogram_fMEP, time_per_step)):
						# raw data before filtering
						peaks_time = np.array(peak_time_per_slice) * dstep_to
						peaks_value = np.array(slice_data)[peak_time_per_slice] + slice_index
						# get peaks only inside the borders
						filter_mask = (border[0] <= peaks_time) & (peaks_time <= border[1])
						# filter data
						peaks_time = peaks_time[filter_mask]
						peaks_value = peaks_value[filter_mask]
						# plot peaks if not void
						if len(peaks_time):
							passed += 1
							sliced_x[slice_index] += list(peaks_time)
							plt.plot(peaks_time, peaks_value, '.', c='k', zorder=3, ms=4)
				# plot boxplots
				for i, x in enumerate(sliced_x):
					if len(x):
						bx = plt.boxplot(x, positions=[i], widths=0.8, whis=[10, 90],
						                 showfliers=False, patch_artist=True, vert=False, zorder=5)
						starts = bx['whiskers'][0].get_xdata()[1]
						plt.text(x=starts - 1.5, y=i + 0.2, s=f"{starts:.1f}", fontsize=25)
						self._recolor(bx, color="#287a72", fill_color="#287a72", fill_alpha=0.2)
				logging.info(f"{shortname}, rat {rat_id}, {passed / alles * 100:.1f}% of peaks prob. at {border}ms")

			save_filename = f"{shortname}_{rat_id}_fMEP_boxplot"
			plt.grid(which='both', axis='x')
			self.axis_article_style(ax, axis='x')
			plt.yticks(range(0, total_slices), self._form_ticklabels(total_slices), fontsize=30)
			plt.xlim(0, slice_in_ms)
			plt.tight_layout()
			plt.savefig(f"{self.plots_folder}/{save_filename}.pdf", format="pdf")
			if show:
				plt.show()
			plt.close()

	@staticmethod
	def joint_plot(X, Y, ax, gs, borders, **kwargs):
		"""
		TODO: add docstring
		Args:
			X (np.ndarray):
			Y (np.ndarray):
			ax:
			gs:
			borders:
			**kwargs:
		"""
		color = kwargs['color']
		xmin, xmax, ymin, ymax = borders

		if kwargs['with_boxplot']:
			pos = kwargs['pos']
			# create X-marginal (top)
			ax_top = plt.subplot(gs[0, 1])
			ax_top.spines['top'].set_visible(False)
			ax_top.spines['right'].set_visible(False)
			# create Y-marginal (right)
			ax_right = plt.subplot(gs[1, 2])
			ax_right.spines['top'].set_visible(False)
			ax_right.spines['right'].set_visible(False)

			ax_left = plt.subplot(gs[1, 0])
			ax_left.spines['top'].set_visible(False)
			ax_left.spines['right'].set_visible(False)

			ax_bottom = plt.subplot(gs[2, 1])
			ax_bottom.spines['top'].set_visible(False)
			ax_bottom.spines['right'].set_visible(False)

			flierprops = dict(marker='.', markersize=1, linestyle='none')

			bxt = ax_left.boxplot(Y, positions=[pos * 10], widths=3, patch_artist=True, flierprops=flierprops)
			recolor(bxt, 'k', color)
			ax_left.set_ylim([ymin, ymax])
			ax_left.set_xticks([])

			bxt = ax_bottom.boxplot(X, positions=[pos], widths=0.4, vert=False, patch_artist=True,
			                        flierprops=flierprops)
			recolor(bxt, 'k', color)
			ax_bottom.set_xlim([xmin, xmax])
			ax_bottom.set_yticks([])
		else:
			ax_top = plt.subplot(gs[0, 0])
			ax_top.spines['top'].set_visible(False)
			ax_top.spines['right'].set_visible(False)
			# create Y-marginal (right)
			ax_right = plt.subplot(gs[1, 1])
			ax_right.spines['top'].set_visible(False)
			ax_right.spines['right'].set_visible(False)

		# add grid
		ax_top.grid(which='minor', axis='x')
		ax_right.grid(which='minor', axis='y')
		# gaussian_kde calculation
		xx = np.linspace(xmin, xmax, 100)
		yy = np.linspace(ymin, ymax, 100)
		dx = st.gaussian_kde(X)(xx)
		dy = st.gaussian_kde(Y)(yy)
		ax_top.plot(xx, dx, color=color)
		ax_right.plot(dy, yy, color=color)
		# plt.plot([0], [kwargs['k']], 'D', color=color, ms=10)
		ax_top.set_yticks([])
		ax_right.set_xticks([])

		return ax_top, ax_right

	def plot_mono_Hz(self, source, rats):
		""""""
		if type(source) is not Metadata:
			if type(source) is dict:
				metadata = Metadata(sdict=source)
			else:
				metadata = Metadata(self.pickle_folder, source)
		else:
			metadata = source

		if rats is None or rats is all:
			rats = metadata.get_rats_id()
		if type(rats) is int:
			rats = [rats]

		shortname = metadata.shortname
		print(shortname)
		#
		slice_length = 50
		ees = int(1 / metadata.rate * 1000)
		# process each rat's data
		for rat_id in rats:
			print(f" rat ID {rat_id} (MONO)".center(30, "-"))
			T = metadata.get_peak_times(rat_id, muscle='E', flat=True) * metadata.dstep_to
			A = metadata.get_peak_ampls(rat_id, muscle='E', flat=True)
			S = metadata.get_peak_slices(rat_id, muscle='E', flat=True)
			D = metadata.get_myograms(rat_id, muscle='E')
			# collect peaks' amplitudes which located inside of mono
			monos = []
			# plot myogram
			colors = iter(["#275b78", "#f2aa2e", "#a6261d", "#472650"] * 100)

			for exp in D:
				for i, data in enumerate(exp):
					color = next(colors)
					plt.plot(np.arange(len(data)) * metadata.dstep_to, data + i, color=color)
			plt.plot(T, S, '.', c='k')
			# process each mono after EES
			for i in range(0, slice_length, ees):
				start = i + 3
				end = i + 7.5
				mask_inside_mono = (start <= T) & (T <= end)
				monos.append(A[mask_inside_mono])
				x, y = T[mask_inside_mono], S[mask_inside_mono]
				plt.axvspan(xmin=start, xmax=end, alpha=0.5)
				plt.plot(x, y, '.', color='r', ms=10)
			# show ratio of average ampls
			for i in range(1, len(monos)):
				print(f"mono #{i} with #0: avg ampls ratio "
				      f"{np.median(monos[i]) / np.median(monos[0]):.3f}\t"
				      f"({np.median(monos[i]):.3f} / {np.median(monos[0]):.3f})")
			plt.show()

	@staticmethod
	def is_inside(points, rc, rx, ry, angle=0):
		cos_angle = np.cos(np.radians(180 - angle))
		sin_angle = np.sin(np.radians(180 - angle))

		xc = points[:, 0] - rc[0]
		yc = points[:, 1] - rc[1]

		xct = xc * cos_angle - yc * sin_angle
		yct = xc * sin_angle + yc * cos_angle

		rad_cc = (xct ** 2 / rx ** 2) + (yct ** 2 / ry ** 2)
		return rad_cc <= 1

	def ellipse_form(self, meta_ellipse):
		"""
		create a shapely ellipse. adapted from
		https://gis.stackexchange.com/a/243462
		"""
		ell_c, ell_w, ell_h, ell_angle = meta_ellipse
		# create ellipse
		circ = Point(ell_c).buffer(1)
		ell = affinity.scale(circ, ell_w, ell_h)
		ellipse = affinity.rotate(ell, ell_angle)
		# form polygon for drawing
		verts = np.array(ellipse.exterior.coords.xy)
		patch = Polygon(verts.T, alpha=0.5)
		return ellipse, patch

	def plot_poly_Hz(self, source, rats):
		""""""
		if type(source) is not Metadata:
			if type(source) is dict:
				metadata = Metadata(sdict=source)
			else:
				metadata = Metadata(self.pickle_folder, source)
		else:
			metadata = source

		if rats is None or rats is all:
			rats = metadata.get_rats_id()
		if type(rats) is int:
			rats = [rats]

		shortname = metadata.shortname
		ell_width = 25
		ell_height = 6
		# ellipse form
		rx = ell_width / 2
		ry = ell_height / 2
		print(shortname)
		slice_length = 50
		ees = int(1 / metadata.rate * 1000)
		#
		for rat_id in rats:
			print(f" rat ID {rat_id} (POLY)".center(30, "-"))
			ellipses = []
			#
			plt.figure(figsize=(10, 5))
			T = metadata.get_peak_times(rat_id, muscle='E', flat=True)
			A = metadata.get_peak_ampls(rat_id, muscle='E', flat=True)
			S = metadata.get_peak_slices(rat_id, muscle='E', flat=True)
			D = metadata.get_myograms(rat_id, muscle='E')

			# process ellipses after each EES
			for i in range(0, slice_length, ees):
				print(i)
				mono_start = i + 3
				mono_end = i + 7
				rc = (mono_end + 1 + ell_width / 2, 2.5)
				print(mono_end)
				# find peaks (time, slice index, ampl) inside ellipse
				points = np.vstack((T * metadata.dstep_to, S, A)).T
				mask_inside = self.is_inside(points, rc=rc, rx=rx, ry=ry)
				points = points[mask_inside]
				ampls = A[mask_inside]
				plt.axvspan(xmin=mono_start, xmax=mono_end, alpha=0.5, color='#472650')

				# remove points inside mono answers
				for time in range(0, slice_length, ees):
					mask_outside_mono = (points[:, 0] < (time + 3)) | ((time + 7.5) < points[:, 0])
					points = points[mask_outside_mono]
					ampls = ampls[mask_outside_mono]
				if len(points) == 0:
					continue
				# ell = Ellipse(xy=rc, width=rx * 2, height=ry * 2, alpha=0.3, edgecolor='k')
				# plt.gca().add_artist(ell)
				ellipses.append((rc, rx, ry, 0, points, ampls))
				print(f"Ellipse #{int(i / ees)} {rc, rx, ry, 0} ampls avg: {np.mean(ampls):.3f}")
				# plot mono area
				print(mono_end)
				plt.plot(points[:, 0], points[:, 1], '.', c='#a6261d', ms=10)

			if len(ellipses) == 1:
				ellipse, patch = self.ellipse_form(ellipses[0][:4])
				plt.gca().add_patch(patch)
			else:
				for i in range(len(ellipses) - 1):
					# first ellipse
					meta_ell1 = ellipses[i]
					ell_polygon1, patch1 = self.ellipse_form(meta_ell1[:4])
					plt.gca().add_patch(patch1)
					# second ellipse
					meta_ell2 = ellipses[i + 1]
					ell_polygon2, patch2 = self.ellipse_form(meta_ell2[:4])
					plt.gca().add_patch(patch2)
					# intersect
					intersect = ell_polygon1.intersection(ell_polygon2)
					if intersect:
						verts3 = np.array(intersect.exterior.coords.xy)
						patch3 = Polygon(verts3.T, facecolor='none', edgecolor='black')
						plt.gca().add_patch(patch3)
						mask_common = (meta_ell1[4][:, None] == meta_ell2[4]).all(-1).any(-1)
						avg_ampls = np.mean(meta_ell1[5][mask_common])
						print(f'area of intersect (#{i} and #{i + 1}): {intersect.area:.3f}, avg ampl in intersect: {avg_ampls:.3f}')
			# just plot all peaks

			colors = iter(["#275b78", "#f2aa2e", "#a6261d", "#472650", "#287a72"] * 100)

			for exp, texp in zip(D, metadata.get_peak_times(rat_id, muscle='E')):
				for islice, (data, tdata) in enumerate(zip(exp, texp)):
					color = next(colors)
					plt.plot(np.arange(len(data)) * metadata.dstep_to, data + islice, color=color)
					if tdata:
						tdata = np.array(tdata)
						plt.plot(tdata * metadata.dstep_to, data[tdata] + islice, '.', c='k', ms=4, zorder=4)

			plt.xlim(0, 50)
			plt.ylim(-1, 6)
			plt.tight_layout()
			#plt.show()
			save_filename = f"{shortname}_hzs.pdf"
			plt.savefig(f"{self.plots_folder}/{save_filename}", dpi=250, format="pdf")
			plt.close()

	def plot_density_3D(self, source, rats, factor=8, show=False, only_volume=False, slice_ms=25):
		"""

		Args:
			source:
			rats (int or tuple):
			factor:
			show:
			only_volume:

		Returns:

		"""
		if type(source) is not Metadata:
			if type(source) is dict:
				metadata = Metadata(sdict=source)
			else:
				metadata = Metadata(self.pickle_folder, source)
		else:
			metadata = source

		if rats is None or rats is all:
			rats = metadata.get_rats_id()
		if type(rats) is int:
			rats = [rats]

		shortname = metadata.shortname

		volumes = []
		#
		for rat_id in rats:
			X = metadata.get_peak_ampls(rat_id, muscle='F', flat=True)
			Y = metadata.get_peak_slices(rat_id, muscle='F', flat=True)
			times = metadata.get_peak_times(rat_id, muscle='E', flat=True) * metadata.dstep_to
			# mask = (times <= 3) | (8 <= times)
			#
			# X = X[mask]
			# Y = Y[mask]

			save_filename = f"{shortname}_3D_rat={rat_id}"
			# form a mesh grid
			xmax, ymax = 1, max(Y)
			xborder_l, xborder_r = 0, 1

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
			if any(n in shortname for n in ["AIR", "TOE", "PLT"]):
				z_mid = (np.max(z) + np.min(z)) / 2
			else:
				z_mid = (np.max(z) + np.min(z)) / 3 * 2

			conty_ymax = -np.inf
			conty_ymin = np.inf

			for i, cont in enumerate(plt.contour(xmesh, ymesh, z, levels=10, alpha=0).allsegs[::-1]):
				if cont:
					contour = max(cont, key=np.size)
					for islice in range(ymax + 1):
						mask = contour[:, 1].astype(int) == islice
						if any(mask):
							# print(f"slice {islice}, ampl = {contour[mask, 0][-1]}")
							print(f"{contour[mask, 0][-1]:.3f}", end='\t')
						else:
							print(0, end='\t')
					print()
				else:
					print("empty contour")
			print("=====")

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
			data = [surface]#, plane1, plane2, plane3, plane4]
			# plot isoline
			for contour in mid_contours:
				data.append(go.Scatter3d(x=contour[:, 0], y=contour[:, 1], z=[z_mid] * len(contour[:, 0]),
				                         line=dict(color='#000000', width=6), mode='lines', showlegend=False))
			# plot dots under isoline
			data.append(go.Scatter3d(x=xmesh[xslice, yslice][zclip <= z_mid].ravel(), # X under isoline
			                         y=ymesh[xslice, yslice][zclip <= z_mid].ravel(), # Y under isoline
			                         z=zunder.ravel(), # Z under isoline
			                         mode='markers', marker=dict(size=2, color=['rgb(0,0,0)'] * len(zunder.ravel()))))
			# plot all
			fig = go.Figure(data=data)
			# change a camera view and etc
			fig.update_layout(title=f'{shortname} | RAT {rat_id} | V: {zvol:.3f}',
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

	def plot_shadow_slices(self, source, rats=None, only_extensor=False, add_kde=False, show=False):
		shadow_color = "#472650"
		kde_color = "#275b78"
		k_fliers_high, k_fliers_low = 5, 6

		if type(source) is not Metadata:
			metadata = Metadata(self.pickle_folder, source)
		else:
			metadata = source

		if rats is None or rats is all:
			rats = metadata.get_rats_id()
		if type(rats) is int:
			rats = [rats]

		shortname = metadata.shortname
		dstep_to = metadata.dstep_to
		speed = metadata.speed
		slice_in_ms = 1 / 40 * 1000

		if speed == "6":
			figsize = (20, 20)
		elif speed == "13.5":
			figsize = (16, 12)
		else:
			figsize = (16, 8)
		#
		for rat_id in rats:
			extensor_data = metadata.get_myograms(rat_id, muscle='E')
			# check rat's flexor, in some cases there are no data
			flexor_flag = rat_id in metadata.get_rats_id(muscle='F') and not only_extensor
			# get number of slices per muscle
			e_slices_number = len(extensor_data[0])
			steps_in_slice = len(extensor_data[0][0])
			# calc boxplots of original data ()
			e_boxplots = get_boxplots(extensor_data)
			# combine data into one list

			plt.close('all')
			fig, ax = plt.subplots(figsize=figsize)

			yticks = []
			f_slices_number = 0 # init flexor number of slices
			shared_x = np.arange(steps_in_slice) * dstep_to
			# plot extensor
			for slice_index, data in enumerate(e_boxplots):
				# set ideal or median
				ideal_data = extensor_data[0][slice_index] + slice_index
				data += slice_index
				# fliers shadow
				ax.fill_between(shared_x, data[:, k_fliers_high], data[:, k_fliers_low],
				                color=shadow_color, alpha=0.7, zorder=3)
				# ideal pattern
				ax.plot(shared_x, ideal_data, color='k', linewidth=2, zorder=4)
				yticks.append(ideal_data[0])

			if flexor_flag:
				flexor_data = metadata.get_myograms(rat_id, muscle='F')
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
					ax.plot(shared_x, ideal_data, color='k', linewidth=2, zorder=4)
					yticks.append(ideal_data[0])

			if add_kde:
				x = metadata.get_peak_times(rat_id, muscle='E', flat=True) * dstep_to
				y = metadata.get_peak_slices(rat_id, muscle='E', flat=True)
				borders = 0, slice_in_ms, -1, e_slices_number
				self._contour_plot(x=x, y=y, color=kde_color, ax=ax, z_prev=[0], borders=borders, levels_num=15, addtan=False)
				if flexor_flag:
					'''flexor'''
					x = metadata.get_peak_times(rat_id, muscle='F', flat=True) * dstep_to
					y = metadata.get_peak_slices(rat_id, muscle='F', flat=True) + e_slices_number
					borders = 0, slice_in_ms, e_slices_number, e_slices_number + f_slices_number
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

	def _get_peaks(self, sliced_datasets, dstep, borders, filter_val, tails=False, debug=False):
		"""
		Finds all peaks times and amplitudes at each slice
		Args:
			sliced_datasets (np.ndarray):
			dstep (float): data step size
			borders (list): time borders for searching peaks
			filter_val (float): default is 0.028 but can be changed
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
		min_dist = int(0.2 / dstep) # 0.15
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
				if (min_dist <= dT <= max_dist) and dA >= filter_val or dA >= min_ampl:
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
	def _multi_R_KDE_test(x1, y1, x2, y2):
		r_fct_string = """
		KDE_test <- function(X1, Y1, X2, Y2){
			library("ks")
			if(length(dim(X1)) == 1){
				X1 <- as.vector(X1)
				X2 <- as.vector(X2)
				Y1 <- as.vector(Y1)
				Y2 <- as.vector(Y2)
				res_time <- kde.test(x1=X1, x2=X2)$pvalue
				res_ampl <- kde.test(x1=Y1, x2=Y2)$pvalue
				mat1 <- matrix(c(X1, Y1), nrow=length(X1))
				mat2 <- matrix(c(X2, Y2), nrow=length(X2))
				res_2d <- kde.test(x1=mat1, x2=mat2)$pvalue
				return(c(res_time, res_ampl, res_2d))
			}
			results <- matrix(, nrow = nrow(X1) * nrow(X2), ncol = 3)
			index <- 1
			#
			for(i1 in 1:nrow(X1)) {
				#
				x1 <- X1[i1, ]
				x1 <- x1[x1 >= 0]
				y1 <- Y1[i1, ]
				y1 <- y1[y1 >= 0]
				#
				for(i2 in 1:nrow(X2)) {
					#
					x2 <- X2[i2, ]
					x2 <- x2[x2 >= 0]
					y2 <- Y2[i2, ]
					y2 <- y2[y2 >= 0]
					#
					mat1 <- matrix(c(x1, y1), nrow=length(x1))
					mat2 <- matrix(c(x2, y2), nrow=length(x2))
					#
					res_time <- kde.test(x1=x1, x2=x2)$pvalue
					res_ampl <- kde.test(x1=y1, x2=y2)$pvalue
					res_2d <- kde.test(x1=mat1, x2=mat2)$pvalue

					results[index, ] <- c(res_time, res_ampl, res_2d)
					index <- index + 1
				}
			}
			return(results)
		}
		"""
		r_pkg = STAP(r_fct_string, "r_pkg")
		rx1, ry1, rx2, ry2 = map(numpy2rpy, (x1, y1, x2, y2))
		return np.asarray(r_pkg.KDE_test(rx1, ry1, rx2, ry2))

	@staticmethod
	def _contour_plot(x, y, color, ax, z_prev, borders, levels_num, addtan = False):
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
		ax.contour(xx, yy, z, levels=levels, linewidths=1, colors=color)
		z_mid = (np.max(z) + np.min(z)) / 2.3
		mid_contours = plt.contour(xx, yy, z, levels=[z_mid], alpha=0).allsegs[0]
		for contour in mid_contours:
			plt.plot(contour[:, 0], contour[:, 1], c='#f2aa2e', linewidth=4)


		if addtan:
			max_contour = max(mid_contours, key=np.size)

			unique, index = np.unique(max_contour[:, 0], axis=0, return_index=True)
			x = np.array(max_contour[:, 0])[index]
			y = np.array(max_contour[:, 1])[index]

			ind = np.lexsort((y,x))
			sorted_x = np.array([x[i] for i in ind])
			sorted_y = np.array([y[i] for i in ind])
			print(sorted_x)
			print(sorted_y)

			mask = ((sorted_x >= 18) & (sorted_x <= 20) & (sorted_y < 21))
			masked_x = sorted_x[mask]
			masked_y = sorted_y[mask]
			print(masked_x)
			print(masked_y)

			t,c,k = interpolate.splrep(masked_x, masked_y, k=3)
			b = interpolate.BSpline(t, c, k)
			fsec = b.derivative(nu=2)
			# fsec = interpolate.splev(24.4, spl, der=2)
			print(fsec)
			# print(interpolate.sproot((t, c - fsec, k)))

			pointcur = interpolate.sproot((fsec.t, fsec.c, k))[-2]
			print(interpolate.sproot((fsec.t, fsec.c, k)))
			spl = interpolate.splrep(sorted_x, sorted_y, k=1)
			small_t = np.arange(pointcur - 8.8, pointcur + 2.7, 0.1)
			print(small_t)
			t,c,k = interpolate.splrep(masked_x, masked_y, k=1)
			fa = interpolate.splev(pointcur, (t,c,k), der=0)     # f(a)
			print(fa)
			fprime = interpolate.splev(pointcur, (t,c,k), der=1) # f'(a)

			tan = fa + fprime * (small_t - pointcur) # tangent
			print(tan)
			slopedegree = math.atan2((small_t[-1] - small_t[0]), (tan[-1] - tan[0])) * 180 / math.pi
			print(f'SLOPE IN DEGREE - {slopedegree}')
			plt.plot(small_t, tan, c='#a6261d', linewidth=5)
			plt.plot(pointcur, fa, 'om')

		ax.contourf(xx, yy, z, levels=levels, colors=colors, alpha=0.7, zorder=0)
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
