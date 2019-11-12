import os
import numpy as np
import pylab as plt
from itertools import chain
from matplotlib import gridspec
from scipy.stats import ks_2samp
from analysis.functions import peacock2
from analysis.functions import auto_prepare_data
from analysis.PCA import joint_plot, contour_plot, get_all_peak_amp_per_slice


def ks_data_test(filepath_a, filepath_b, debugging=False):
	"""
	Function for getting data from files by filepath, preparing them and comparing by K-S test
	Args:
		filepath_a (str): filepath to hdf5
		filepath_b (str): filepath to hdf5
	"""
	dstep_to = 0.1
	peaks_pack = []
	ampls_pack = []
	names_pack = []
	colors = ("#FE7568", "#315B8A")
	flatten = chain.from_iterable

	for filepath in [filepath_a, filepath_b]:
		if not os.path.exists(filepath_a):
			raise FileNotFoundError

		folder = os.path.dirname(filepath)
		filename = os.path.basename(filepath)

		# get extensor/flexor prepared data (centered, normalized, subsampled and sliced)
		prepared_data = auto_prepare_data(folder, filename, dstep_to=dstep_to)

		# for 1D or 2D Kolmogorod-Smirnov test (without pattern)
		e_peak_times_per_slice, e_peak_ampls_per_slice = get_all_peak_amp_per_slice(prepared_data, dstep_to)
		times = np.array(list(flatten(flatten(e_peak_times_per_slice)))) * dstep_to
		ampls = np.array(list(flatten(flatten(e_peak_ampls_per_slice))))

		peaks_pack.append(times)
		ampls_pack.append(ampls)
		names_pack.append(filename)

		print(filename)

	x1, x2 = peaks_pack
	y1, y2 = ampls_pack

	dvalue, pvalue = ks_2samp(x1, x2)
	print(f"1D K-S peaks TIME -> D-value: {dvalue}, p-value: {pvalue}")

	dvalue, pvalue = ks_2samp(y1, y2)
	print(f"1D K-S peaks AMPLITUDE -> D-value: {dvalue}, p-value: {pvalue}")

	d1 = np.stack((x1, y1), axis=1)
	d2 = np.stack((x2, y2), axis=1)

	dvalue, pvalue = peacock2(d1, d2)
	print(f"2D peacock TIME/AMPLITUDE -> D-value: {dvalue}, p-value: {pvalue}")

	if debugging:
		# plot 2D
		# define grid for subplots
		gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])
		fig = plt.figure()
		kde_ax = plt.subplot(gs[1, 0])
		kde_ax.spines['top'].set_visible(False)
		kde_ax.spines['right'].set_visible(False)

		for x, y, name, color in zip(peaks_pack, ampls_pack, names_pack, colors):
			contour_plot(x=x, y=y, color=color, ax=kde_ax)
			joint_plot(x, y, kde_ax, gs, **{"color": color})

		kde_ax.set_xlabel("peak time (ms)")
		kde_ax.set_ylabel("peak amplitude")
		kde_ax.set_xlim(0, 25)
		kde_ax.set_ylim(0, 2)
		plt.suptitle("\n".join(names_pack))
		plt.tight_layout()
		plt.show()
		plt.close(fig)


if __name__ == "__main__":
	filepath1 = "/home/alex/GitHub/DATA/gras/foot/gras_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5"
	filepath2 = "/home/alex/GitHub/DATA/bio/foot/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5"
	ks_data_test(filepath1, filepath2)
