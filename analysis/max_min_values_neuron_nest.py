import numpy as np
import pylab as plt
from analysis.functions import *
from collections import defaultdict
from analysis.real_data_slices import read_data, slice_myogram, plot_1d, plot_by_slice

test_number = 25
nest_sim_step = 0.025
neuron_sim_step = 0.025
real_data_step = 0.25


def read_data_NEST(file_path):
	"""

	Args:
		file_path:

	Returns:

	"""
	tests_nest = {k: {} for k in range(test_number)}
	for test_index in range(test_number):
		nrns_nest = set()
		with open('{}/{}.dat'.format(file_path, test_index), 'r') as file:
			for line in file:
				nrn_id, time, volt = line.split("\t")[:3]
				time = float(time)
				if time not in tests_nest[test_index].keys():
					tests_nest[test_index][time] = 0
				tests_nest[test_index][time] += float(volt)
				nrns_nest.add(nrn_id)
		for time in tests_nest[test_index].keys():
			tests_nest[test_index][time] = round(tests_nest[test_index][time] / len(nrns_nest), 3)
	return tests_nest


def read_data_NEURON(filepath):
	"""

	Args:
		filepath:

	Returns:

	"""
	tests_neuron = {k: {} for k in range(test_number)}
	V_to_uV = 1000000
	neuron_number = 169
	for test_index in range(test_number):
		for neuron_id in range(neuron_number):
			with open('{}/volMN{}v{}.txt'.format(filepath, neuron_id, test_index), 'r') as file:
				for time, value in enumerate([float(i) * V_to_uV for i in file.read().split("\n")[1:-2]]):
					time = round(time * neuron_sim_step, 3)
					if time not in tests_neuron[test_index].keys():
						tests_neuron[test_index][time] = 0
					tests_neuron[test_index][time] += (value * 10 ** 7) / neuron_number
	return tests_neuron


def calc_NEURON(data, debug_show=False):
	"""

	Args:
		data:
		debug_show:

	Returns:

	"""
	neuron_tests = []
	offset = 1000
	prev_additional_step = 1

	for k, v in data.items():
		neuron_tests.append(list(v.values()))

	extremum_max_points_times = {k: defaultdict(list) for k in range(len(neuron_tests))}
	extremum_min_points_times = {k: defaultdict(list) for k in range(len(neuron_tests))}
	extremum_max_points_values = {k: defaultdict(list) for k in range(len(neuron_tests))}
	extremum_min_points_values = {k: defaultdict(list) for k in range(len(neuron_tests))}

	# Find EES as start point for slicing
	slice_begin_ticks = defaultdict(list)
	for test_index, test_data in enumerate(neuron_tests):
		for begin in range(len(test_data))[::offset]:
			EES_value = max(test_data[begin + 140:begin + 200])  # remove over 70 ms to find EES
			slice_begin_ticks[test_index].append(test_data.index(EES_value))
	for test_index, test_data in enumerate(neuron_tests):
		start = slice_begin_ticks[0][0] - prev_additional_step
		for slice_index in range(len(slice_begin_ticks[0])):
			tmp_max_time = []
			tmp_min_time = []
			tmp_max_value = []
			tmp_min_value = []
			sliced_values = test_data[start:start + offset]
			datas_times = range(offset)

			# search min/max
			for i in range(1, len(sliced_values) - 1):
				if sliced_values[i - 1] < sliced_values[i] > sliced_values[i + 1]:
					tmp_max_time.append(round(datas_times[i] * neuron_sim_step, 3))  # with normalization to 1 ms
					tmp_max_value.append(sliced_values[i])
				if sliced_values[i - 1] > sliced_values[i] < sliced_values[i + 1]:
					tmp_min_time.append(round(datas_times[i] * neuron_sim_step, 3))  # with normalization to 1 ms
					tmp_min_value.append(sliced_values[i])

			extremum_max_points_times[test_index][slice_index] = list(tmp_max_time)
			extremum_max_points_values[test_index][slice_index] = list(tmp_max_value)

			extremum_min_points_times[test_index][slice_index] = list(tmp_min_time)
			extremum_min_points_values[test_index][slice_index] = list(tmp_min_value)

			start += offset

	print("NEURON")
	for test_index, test_data in extremum_max_points_times.items():
		print(test_index)
		for slice_index, slice_data in test_data.items():
			print("\t", slice_index, *slice_data)

	if debug_show:
		for test_index in range(len(extremum_max_points_times)):
			plt.figure()
			plt.plot([x for x in range(len(neuron_tests[test_index]))], neuron_tests[test_index], color="gray")
			for slice_index, _ in extremum_max_points_times[test_index].items():
				x = extremum_max_points_times[test_index][slice_index]
				y = extremum_max_points_values[test_index][slice_index]
				plt.plot(
					[o / neuron_sim_step + offset * slice_index + (slice_begin_ticks[0][0] - prev_additional_step) for o
					 in x], y, ".", color="r")
			for slice_index, _ in extremum_min_points_times[test_index].items():
				x = extremum_min_points_times[test_index][slice_index]
				y = extremum_min_points_values[test_index][slice_index]
				plt.plot(
					[o / neuron_sim_step + offset * slice_index + (slice_begin_ticks[0][0] - prev_additional_step) for o
					 in x], y, ".", color="b")
			plt.show()
		plt.close()

	return extremum_max_points_times, extremum_min_points_times


def calc_NEST(data, debug_show=False):
	"""

	Args:
		data:
		debug_show:

	Returns:

	"""
	nest_tests = []
	offset = 250

	for k, v in data.items():
		nest_tests.append(list(v.values()))

	extremum_max_points_times = {k: defaultdict(list) for k in range(len(nest_tests))}
	extremum_min_points_times = {k: defaultdict(list) for k in range(len(nest_tests))}
	extremum_max_points_values = {k: defaultdict(list) for k in range(len(nest_tests))}
	extremum_min_points_values = {k: defaultdict(list) for k in range(len(nest_tests))}

	# Find EES as start point for slicing
	slice_begin_ticks = defaultdict(list)
	for test_index, test_data in enumerate(nest_tests):
		for begin in range(len(test_data))[::offset]:
			EES_value = min(test_data[begin:begin + 70])  # remove over 70 ms to find EES
			slice_begin_ticks[test_index].append(test_data.index(EES_value))

	for test_index, test_data in enumerate(nest_tests):
		start = slice_begin_ticks[0][0] - 10  # remove 10 if you want calculate without EES
		for slice_index in range(len(slice_begin_ticks[0])):
			tmp_max_time = []
			tmp_min_time = []
			tmp_max_value = []
			tmp_min_value = []
			sliced_values = test_data[start:start + offset]
			datas_times = range(offset)

			# search min/max
			for i in range(1, len(sliced_values) - 1):
				if sliced_values[i - 1] < sliced_values[i] > sliced_values[i + 1]:
					tmp_max_time.append(round(datas_times[i] * nest_sim_step, 3))  # with normalization to 1 ms
					tmp_max_value.append(sliced_values[i])
				if sliced_values[i - 1] > sliced_values[i] < sliced_values[i + 1]:
					tmp_min_time.append(round(datas_times[i] * nest_sim_step, 3))  # with normalization to 1 ms
					tmp_min_value.append(sliced_values[i])

			extremum_max_points_times[test_index][slice_index] = list(tmp_max_time)
			extremum_max_points_values[test_index][slice_index] = list(tmp_max_value)

			extremum_min_points_times[test_index][slice_index] = list(tmp_min_time)
			extremum_min_points_values[test_index][slice_index] = list(tmp_min_value)

			start += offset

	print("NEST")
	for test_index, test_data in extremum_max_points_times.items():
		print(test_index)
		for slice_index, slice_data in test_data.items():
			print("\t", slice_index, *slice_data)

	if debug_show:
		for test_index in range(len(extremum_max_points_times)):
			plt.figure()
			plt.plot([x for x in range(len(nest_tests[test_index]))], nest_tests[test_index], color="gray")
			for slice_index, _ in extremum_max_points_times[test_index].items():
				x = extremum_max_points_times[test_index][slice_index]
				y = extremum_max_points_values[test_index][slice_index]
				plt.plot([o / nest_sim_step + offset * slice_index + (slice_begin_ticks[0][0] - 10) for o in x], y, ".",
				         color="r")
			for slice_index, _ in extremum_min_points_times[test_index].items():
				x = extremum_min_points_times[test_index][slice_index]
				y = extremum_min_points_values[test_index][slice_index]
				plt.plot([o / nest_sim_step + offset * slice_index + (slice_begin_ticks[0][0] - 10) for o in x], y, ".",
				         color="b")
			plt.show()
		plt.close()

	return extremum_max_points_times, extremum_min_points_times


def calc_real_data(volt_data, slices_begin_time, debug_show=False):
	"""

	Args:
		volt_data:
		slices_begin_time:
		debug_show:

	Returns:

	"""
	slices_begin_time = [int(t / real_data_step) for t in slices_begin_time]
	real_max_min = calc_max_min(slices_begin_time, volt_data, real_data_step)
	print("slices_begin_time", slices_begin_time)
	print("volt_data", volt_data)
	print("real_data_step", real_data_step)
	slices_max_time = real_max_min[0]
	slices_max_value = real_max_min[1]
	slices_min_time = real_max_min[2]

	slices_min_value = real_max_min[3]
	print("REAL data")
	print("MAX")
	for slice_index, times in slices_max_time.items():
		print("\t", slice_index, *times)
	print("MIN")
	for slice_index, times in slices_min_time.items():
		print("\t", slice_index, *times)

	volt_data = volt_data[slices_begin_time[10]:slices_begin_time[16]]
	if debug_show:
		plt.figure()
		plt.suptitle("REAL 40Hz (slow) Extensor (0.25 ms step)")
		plt.plot([x / 4 for x in range(len(volt_data))], volt_data, color="gray")
		for slice_num in range(0, 6):
			plt.plot([(x + slice_num * 25) for x in slices_max_time[slice_num + 1]],
			         slices_max_value[slice_num + 1], ".", color='red')
			plt.plot([(x + slice_num * 25) for x in slices_min_time[slice_num + 1]],
			         slices_min_value[slice_num + 1], ".", color='blue')
		plt.xlim(0, 150)
		plt.show()
		plt.close()

	return slices_max_time, slices_min_time


def plot(real_results, NEST_results, NEURON_results):
	"""

	Args:
		real_results:
		NEST_results:
		NEURON_results:

	Returns:

	"""
	for simulator_index, extremums_data in enumerate([NEST_results, NEURON_results]):
		simulator_name = "NEST" if simulator_index == 0 else "NEURON"
		for index_element, extremum in enumerate([extremums_data[0], extremums_data[1]]):
			lavrov_extremums = list(real_results[index_element].values())
			compare_by = "MAX" if index_element == 0 else "MIN"
			diffs = {k: defaultdict(list) for k in range(6)}
			for slice_index in extremum[0].keys():
				for test_index in extremum.keys():
					# find neighbour dots Lavrov <-> simulation
					for point_index in range(len((lavrov_extremums[slice_index]))):
						curr_point_Lavrov = lavrov_extremums[slice_index][point_index]
						prev_point_Lavrov = lavrov_extremums[slice_index][point_index - 1] \
							if point_index - 1 >= 0 else 0
						next_point_Lavrov = lavrov_extremums[slice_index][point_index + 1] \
							if point_index + 1 < len(lavrov_extremums[slice_index]) else 25

						left_window = abs(prev_point_Lavrov - curr_point_Lavrov) / 2
						right_window = abs(next_point_Lavrov - curr_point_Lavrov) / 2
						tmp = []
						for point_simulator in extremum[test_index][slice_index]:
							if curr_point_Lavrov - left_window <= point_simulator <= curr_point_Lavrov + right_window:
								tmp.append(abs(point_simulator - curr_point_Lavrov))
						diffs[slice_index][point_index] += tmp

			# re-procesing data to convert list to tuple of min mean max
			minimals_per_point = []
			means_per_point = []
			maximals_per_point = []
			for slice_index, points in diffs.items():
				tmp_min = []
				tmp_max = []
				tmp_mean = []
				for point_index, diff in points.items():
					if diff:
						tmp_min.append(min(diff))
						tmp_mean.append(np.mean(diff))
						tmp_max.append(max(diff))
					else:
						tmp_min.append(-1)
						tmp_mean.append(-1)
						tmp_max.append(-1)
				minimals_per_point.append(tmp_min)
				means_per_point.append(tmp_mean)
				maximals_per_point.append(tmp_max)

			# Plot
			offset = 0
			plt.figure()
			plt.suptitle("time diff between real data and {} (per {} extremums)".format(simulator_name, compare_by))
			max_height = 0
			for slice_index in range(6):
				length = len(maximals_per_point[slice_index])
				if max(maximals_per_point[slice_index]) > max_height:
					max_height = max(maximals_per_point[slice_index])
				# Draw maximal
				plt.bar([x + offset for x in range(length)],
				        maximals_per_point[slice_index], color='r', label="max" if slice_index == 0 else None)
				# Draw mean
				plt.bar([x + offset for x in range(length)],
				        means_per_point[slice_index], color='k', label="mean" if slice_index == 0 else None)
				# Draw minimal
				plt.bar([x + offset for x in range(length)],
				        minimals_per_point[slice_index], color='b', label="min" if slice_index == 0 else None)
				# Draw slice border
				plt.axvline(x=offset - 0.5, color='gray', linestyle="--")
				for index, elem in enumerate(minimals_per_point[slice_index]):
					if elem == -1:
						plt.text(index + offset - 0.25, 0.25, "NO", fontsize=12)
				offset += len(minimals_per_point[slice_index])
			a = []
			for l in lavrov_extremums:
				a += l
			plt.xticks(range(len(a)), a)
			plt.xlabel("point index")
			plt.ylim(0, max_height + 1)
			plt.ylabel("time diff in ms")
			plt.grid(axis='y')
			plt.legend()
			plt.show()


def main():
	# raw_NEST_data = read_data_NEST('NEST')
	# NEST_results = calc_NEST(raw_NEST_data, debug_show=False)
	#
	# raw_NEURON_data = read_data_NEURON('res2509')
	# NEURON_results = calc_NEURON(raw_NEURON_data, debug_show=False)

	raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
	volt, slices_time = slice_myogram(raw_real_data)
	real_results = calc_real_data(volt, slices_time, debug_show=False)


if __name__ == "__main__":
	main()
