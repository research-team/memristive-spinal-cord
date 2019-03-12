import subprocess
import numpy as np
import h5py as hdf5
import pylab as plt


def run_tests():
	for i in range(20):
		subprocess.call(["/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/matrix_solution/kek", str(i)])


def hdf():
	filename = "20_tests"
	abs_path = "/home/alex/GitHub/memristive-spinal-cord/izhikevichGPU/matrix_solution"
	all_data = []
	with hdf5.File('{}/{}.hdf5'.format(abs_path, filename), 'w') as write_file:
		for i in range(20):
			with open("{}/{}.dat".format(abs_path, i)) as file_read:
				data = file_read.readlines()
				first = -float(data[0])
				test_data = [-float(d) - first for d in data]
				all_data.append(test_data)
			write_file.create_dataset("test_{}".format(i), data=test_data, compression="gzip")

	return all_data


def plot_shadows(data, step=None, debugging=False):
	"""

	Args:
		data (list of list): data per test per point
		step (float): step size of the data for human-read normalization time
		debugging (bool): show debug info
	Returns:
		kawai pictures
	"""
	tests_number = len(data)
	slice_time_length = 25
	slices_number = int(len(data[0]) / slice_time_length * step)
	steps_in_slice = int(slice_time_length / step)

	all_data = list(zip(*data))

	plt.figure(figsize=(16, 9))

	# per slice
	for slice_index in range(slices_number):
		# set offset for Y
		y_offset = slice_index * 40
		# get data for curent slice
		sliced_data = all_data[slice_index * steps_in_slice:(slice_index + 1) * steps_in_slice]
		# calculate fliers, whiskers and medians (thanks to pylab <3)
		tmp_fig = plt.figure()
		boxplot_data = plt.boxplot(sliced_data, showfliers=True, showcaps=True)
		plt.close(tmp_fig)

		medians = boxplot_data['medians']
		whiskers_data = boxplot_data['whiskers']
		fliers = boxplot_data['fliers']

		whiskers_data_high = whiskers_data[1::2]
		whiskers_data_low = whiskers_data[::2]

		if debugging:
			# back to previous data structure
			tmp_ = list(zip(*sliced_data))
			# plot for each test
			for i in range(tests_number):
				# FixMe +1 to move X data to right (check this)
				plt.plot([x + 1 for x in range(len(tmp_[i]))],
				         [y_offset + y for y in tmp_[i]], color='g')

		fliers_x = []
		fliers_y_max = []
		fliers_y_min = []
		# ToDo in progress
		for i in range(len(whiskers_data)):
			if fliers[i]:
				fliers_x.append(i)
				fliers_y.append(medians[i].get_ydata()[0])
			else:

				fliers_x.append(fliers[10].get_xdata())
				fliers_y.append(medians[i].get_ydata()[0])


		# plot fliers shadow (whiskers top or bottom)
		print()
		print(fliers[10].get_ydata())
		# print(*dir(fliers[0]), sep="\n")
		raise Exception

		# plot whiskers shadow (whiskers top or bottom)
		outer_index = 1
		whiskers_x = [whisker.get_xdata()[0] for whisker in whiskers_data_high]
		whiskers_y_high = [y_offset + whisker.get_ydata()[outer_index] for whisker in whiskers_data_high]
		whiskers_y_low = [y_offset + whisker.get_ydata()[outer_index] for whisker in whiskers_data_low]
		# plot them as filling by color
		plt.fill_between(whiskers_x, whiskers_y_low, whiskers_y_high, alpha=0.3, color='r')

		# plot boxes shadow (like a boxes -- top or bottom, but it is still whisker)
		inner_index = 0
		whiskers_x = [whisker.get_xdata()[0] for whisker in whiskers_data_high]
		whiskers_y_high = [y_offset + whisker.get_ydata()[inner_index] for whisker in whiskers_data_high]
		whiskers_y_low = [y_offset + whisker.get_ydata()[inner_index] for whisker in whiskers_data_low]
		# plot them as filling by color
		plt.fill_between(whiskers_x, whiskers_y_low, whiskers_y_high, alpha=0.7, color='r')

		# plot median
		# need to use np.mean for median_x because median is a 2D line in the box (by mean we find the center)
		median_x = [np.mean(median.get_xdata()) for median in medians]
		median_y = [y_offset + median.get_ydata()[0] for median in medians]
		plt.plot(median_x, median_y, color='k')

		# plot stuff
		plt.xticks([], range(26))

	plt.show()


def run():
	# run_tests()
	data = hdf()
	plot_shadows(data, step=0.25)


if __name__ == "__main__":
	run()

