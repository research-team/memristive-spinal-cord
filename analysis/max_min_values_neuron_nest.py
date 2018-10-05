import pylab as plt
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from analysis.real_data_slices import read_data, data_processing

test_number = 3
nest_sim_step = 0.1
neuron_sim_step = 0.025
real_data_step = 0.25


def read_data_NEST(file_path):
	tests_nest = {k: {} for k in range(test_number)}
	for neuron_id in range(test_number):
		nrns_nest = set()
		with open('{}/{}.dat'.format(file_path, neuron_id), 'r') as file:
			for line in file:
				nrn_id, time, volt = line.split("\t")[:3]
				time = float(time)
				if time not in tests_nest[neuron_id].keys():
					tests_nest[neuron_id][time] = 0
				tests_nest[neuron_id][time] += float(volt)
				nrns_nest.add(nrn_id)
		for time in tests_nest[neuron_id].keys():
			tests_nest[neuron_id][time] = round(tests_nest[neuron_id][time] / len(nrns_nest), 3)
	return tests_nest


def read_data_NEURON(filepath):
	neuron_tests = []
	V_to_uV = 1000000
	# ToDo calculate neuron number by uniq file names
	neuron_number = 169
	for neuron_test_number in range(test_number):
		tmp = []
		for neuron_id in range(neuron_number):
			with open('{}/volMN{}v{}.txt'.format(filepath, neuron_id, neuron_test_number), 'r') as file:
				tmp.append([float(i) * V_to_uV for i in file.read().split("\n")[1:-2]])
		neuron_tests.append([elem * 10 ** 7 for elem in list(map(lambda x: np.mean(x), zip(*tmp)))])
	return neuron_tests


def calc_NEURON(tests_data, debug_show=False):
	# calculate mean
	neuron_means = list(map(lambda x: np.mean(x), zip(*tests_data)))
	# calc min max
	offset = 1000
	datas_times = []
	sliced_values = []

	slice_begin_ticks = []
	for begin in range(len(neuron_means))[::offset]:
		max_first = max(neuron_means[begin+140:begin+200]) # 140 and 200 patamushta
		slice_begin_ticks.append(neuron_means.index(max_first))

	# check diff between maxs
	# for index in range(len(slice_begin_ticks) -1):
	# 	print(slice_begin_ticks[index+1] - slice_begin_ticks[index])

	counter = len(slice_begin_ticks)
	k_slice = 1
	slices_max_time = {}
	slices_max_value = {}
	slices_min_time = {}
	slices_min_value = {}
	start = slice_begin_ticks[0] - 1

	for j in range(counter):
		sliced_values += neuron_means[start:start + offset]
		datas_times += range(start, start + offset)
		tmp_max_time = []
		tmp_min_time = []
		tmp_max_value = []
		tmp_min_value = []

		for c in range(1, len(sliced_values) - 1):
			if sliced_values[c - 1] < sliced_values[c] > sliced_values[c + 1]:
				tmp_max_time.append(datas_times[c])
				tmp_max_value.append(sliced_values[c])
			if sliced_values[c - 1] > sliced_values[c] < sliced_values[c + 1]:
				tmp_min_time.append(datas_times[c])
				tmp_min_value.append(sliced_values[c])

		slices_max_time[k_slice] = tmp_max_time
		slices_max_value[k_slice] = tmp_max_value
		slices_min_time[k_slice] = tmp_min_time
		slices_min_value[k_slice] = tmp_min_value
		start += offset
		k_slice += 1
		sliced_values.clear()

	print("NEURON")
	print(len(slices_max_value), "max = ", slices_max_value)
	print(len(slices_max_time), "max_times = ", slices_max_time)
	print(len(slices_min_value), "min = ", slices_min_value)
	print(len(slices_min_time), "min_times = ", slices_min_time)

	if debug_show:
		plt.figure()
		plt.suptitle("NEURON 40Hz (slow) Extensor (0.025 ms step)")
		plt.plot([x / 40 for x in range(len(neuron_means))], neuron_means, color="gray")
		for slice_num in range(0, 6):
			plt.plot([(x + slice_num * 1000) / 40 for x in slices_max_time[slice_num+1]],
			         slices_max_value[slice_num+1], ".", color='red')
			plt.plot([(x + slice_num * 1000) / 40 for x in slices_min_time[slice_num + 1]],
			         slices_min_value[slice_num + 1], ".", color='blue')
		plt.xlim(0,150)
		plt.show()
		plt.close()

	return slice_begin_ticks, slices_max_time, slices_max_value, slices_min_time, slices_min_value


def calc_NEST(data, debug_show=False):
	# calculate mean
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
		start = slice_begin_ticks[0][0] - 10
		for slice_index in range(len(slice_begin_ticks[0])):
			sliced_values = test_data[start:start + offset]
			datas_times = range(offset)

			tmp_max_time = []
			tmp_min_time = []
			tmp_max_value = []
			tmp_min_value = []

			# search min/max
			for i in range(1, len(sliced_values) - 1):
				if sliced_values[i - 1] < sliced_values[i] > sliced_values[i + 1]:
					tmp_max_time.append(datas_times[i])
					tmp_max_value.append(sliced_values[i])
				if sliced_values[i - 1] > sliced_values[i] < sliced_values[i + 1]:
					tmp_min_time.append(datas_times[i])
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
			print("\t", slice_index)
			print("\t\t", *slice_data)

	for test_number in range(len(extremum_max_points_times)):
		plt.figure()
		plt.plot([x for x in range(len(nest_tests[test_number]))],
		         nest_tests[test_number], color="gray")
		for slice_index, _ in extremum_max_points_times[test_number].items():
			x = extremum_max_points_times[test_number][slice_index]
			y = extremum_max_points_values[test_number][slice_index]
			plt.plot([o + offset * slice_index + (slice_begin_ticks[0][0] - 10) for o in x], y, ".", color="r")
		for slice_index, _ in extremum_min_points_times[test_number].items():
			x = extremum_min_points_times[test_number][slice_index]
			y = extremum_min_points_values[test_number][slice_index]
			plt.plot([o + offset * slice_index + (slice_begin_ticks[0][0] - 10) for o in x], y, ".", color="b")
	plt.show()
	raise Exception
	print(len(slices_max_value), "max = ", slices_max_value)
	print(len(slices_max_time), "max_times = ", slices_max_time)
	print(len(slices_min_value), "min = ", slices_min_value)
	print(len(slices_min_time), "min_times = ", slices_min_time)

	if debug_show:
		plt.figure()
		plt.suptitle("NEST 40Hz (slow) Extensor (0.1 ms step)")
		plt.plot([x / 10 for x in range(len(nest_means))], nest_means, color="gray")
		for slice_num in range(0, 6):
			plt.plot([(x + slice_num * offset) / 10 for x in slices_max_time[slice_num+1]],
			         slices_max_value[slice_num+1], ".", color='red')
			plt.plot([(x + slice_num * offset) / 10 for x in slices_min_time[slice_num + 1]],
			         slices_min_value[slice_num + 1], ".", color='blue')
		plt.xlim(0, 150)
		plt.show()
		plt.close()

	return slice_begin_ticks, slices_max_time, slices_max_value, slices_min_time, slices_min_value


def calc_real_data(volt_data, slices_begin_time, debug_show=False):
	# plt.figure()
	# plt.plot(range(len(volt_data)), volt_data, color="gray")
	# for sli in slices_begin_time:
	# 	plt.axvline(x=sli*4)
	# plt.show()
	# plt.close()

	slices_begin_time = [int(t * 4) for t in slices_begin_time]

	# check diff between maxs
	# for index in range(len(slice_begin_ticks) -1):
	# 	print(slice_begin_ticks[index+1] - slice_begin_ticks[index])
	datas_times = []
	sliced_values = []
	counter = len(slices_begin_time[10:16])
	k_slice = 1
	offset = slices_begin_time[1] - slices_begin_time[0]
	print("OFFSET", offset)
	slices_max_time = {}
	slices_max_value = {}
	slices_min_time = {}
	slices_min_value = {}
	start = slices_begin_time[10] - 1

	for j in range(counter):
		sliced_values += volt_data[start:start + offset]
		datas_times += range(start, start + offset)
		tmp_max_time = []
		tmp_min_time = []
		tmp_max_value = []
		tmp_min_value = []

		for c in range(1, len(sliced_values) - 1):
			if sliced_values[c - 1] < sliced_values[c] > sliced_values[c + 1]:
				tmp_max_time.append(datas_times[c] - 1000)
				tmp_max_value.append(sliced_values[c])
			if sliced_values[c - 1] > sliced_values[c] < sliced_values[c + 1]:
				tmp_min_time.append(datas_times[c] - 1000)
				tmp_min_value.append(sliced_values[c])

		slices_max_time[k_slice] = tmp_max_time
		slices_max_value[k_slice] = tmp_max_value
		slices_min_time[k_slice] = tmp_min_time
		slices_min_value[k_slice] = tmp_min_value
		start += offset
		k_slice += 1
		sliced_values.clear()

	print("REAL")
	print(len(slices_max_value), "max = ", slices_max_value)
	print(len(slices_max_time), "max_times = ", slices_max_time)
	print(len(slices_min_value), "min = ", slices_min_value)
	print(len(slices_min_time), "min_times = ", slices_min_time)

	volt_data = volt_data[slices_begin_time[10] : slices_begin_time[16]]
	if debug_show:
		plt.figure()
		plt.suptitle("REAL 40Hz (slow) Extensor (0.25 ms step)")
		plt.plot([x / 4 for x in range(len(volt_data))], volt_data, color="gray")
		for slice_num in range(0, 6):
			plt.plot([(x + slice_num * 100) / 4 for x in slices_max_time[slice_num+1]],
			         slices_max_value[slice_num+1], ".", color='red')
			plt.plot([(x + slice_num * 100) / 4 for x in slices_min_time[slice_num + 1]],
			         slices_min_value[slice_num + 1], ".", color='blue')
		plt.xlim(0, 150)
		plt.show()
		plt.close()

	return slices_begin_time, slices_max_time, slices_max_value, slices_min_time, slices_min_value



def plot(real_results, NEST_results, NEURON_results, compare_by="max"):
	"""
	:param list real_results:
	:param list NEST_results:
	:param list NEURON_results:
	"""
	# Lavrov results from 10 - 15 slices
	if compare_by == "max":
		index = 1
	else:
		index = 3
	lavrov_max = []
	neuron_max = []
	nest_max = []
	for v in real_results[index].values():
		lavrov_max.append([t / 4 for t in v])
	for v in NEURON_results[index].values():
		neuron_max.append([t / 40 for t in v])
	for v in NEST_results[index].values():
		nest_max.append([t / 10 for t in v])
	print("real_results = ", real_results)
	print("lavrov_max = ", lavrov_max)
	print("neuron_max = ", neuron_max)
	print("nest_max = ", nest_max)

	diff_nest_lavrov_per_slice = []
	diff_neuron_lavrov_per_slice = []
	for slice_num in range(6):
		per_point_nest = []
		per_point_neuron = []
		for t_Lavrov in lavrov_max[slice_num]:
			# NEST
			tmp = []
			for t_p in nest_max[slice_num]:
				if t_Lavrov - 5 < t_p < t_Lavrov + 5:
					print("DA ", abs(t_p - t_Lavrov))
					tmp.append(abs(t_p - t_Lavrov))
				print(tmp)
			per_point_nest.append(np.mean(tmp))
			# NEURON
			tmp = []
			for t_p in neuron_max[slice_num]:
				if t_Lavrov - 5 < t_p < t_Lavrov + 5:
					print("DA ", abs(t_p - t_Lavrov))
					tmp.append(abs(t_p - t_Lavrov))
			per_point_neuron.append(np.mean(tmp))

		diff_nest_lavrov_per_slice.append(per_point_nest)
		diff_neuron_lavrov_per_slice.append(per_point_neuron)
		# print("KEK", per_point_neuron)
	print("diff_neuron_lavrov_per_slice = ", diff_neuron_lavrov_per_slice)
	# ONLY NEURON (max)
	start = 0
	plt.figure()
	plt.suptitle("time diff between real data and Neuron ({} values)".format(compare_by))
	for dots_in_slice in diff_neuron_lavrov_per_slice:
		length = len(dots_in_slice)
		x = range(start, start+length)
		plt.bar(x, dots_in_slice)
		start += length
	plt.xticks(range(start), range(start))
	plt.xlabel("point index")
	plt.ylabel("time diff in ms")
	plt.show()

	# ONLY NEST (max)
	start = 0
	plt.figure()
	plt.suptitle("time diff between real data and Nest ({} values)".format(compare_by))
	for dots_in_slice in diff_nest_lavrov_per_slice:
		length = len(dots_in_slice)
		x = range(start, start + length)
		plt.bar(x, dots_in_slice)
		start += length
	plt.xticks(range(start), range(start))
	plt.xlabel("point index")
	plt.ylabel("time diff in ms")
	plt.show()




def main():
	raw_NEST_data = read_data_NEST('NEST')
	NEST_results = calc_NEST(raw_NEST_data, debug_show=False)

	raw_NEURON_data = read_data_NEURON('res2509')
	NEURON_results = calc_NEURON(raw_NEURON_data, debug_show=False)

	raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
	volt, slices_time = data_processing(raw_real_data)
	real_results = calc_real_data(volt, slices_time, debug_show=False)

	plot(real_results, NEST_results, NEURON_results, compare_by="max")

if __name__ == "__main__":
	main()
# datas = {}
# mat_data = sio.loadmat('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')
# tickrate = int(mat_data['tickrate'][0][0])
# datas_max = []
# datas_min = []
# datas_times = []
# datas_max_time = []
# datas_min_time = []
# # Collect data
# for index, data_title in enumerate(mat_data['titles']):
# 	data_start = int(mat_data['datastart'][index])-1
# 	data_end = int(mat_data['dataend'][index])
# 	if "Stim" not in data_title:
# 		datas[data_title] = mat_data['data'][0][data_start:data_end]
# # Plot data
# for data_title, data in datas.items():
# 	x = [i / tickrate for i in range(len(data))]
# 	plt.plot(x, data, label=data_title)
# 	plt.xlim(0, x[-1])
# # if "Stim" not in data_title:
# # 	datas[data_title] = mat_data['data'][0][data_start:data_end]
# values = max(datas.values())
# offset = 100
# sliced_values = []
# start = 188
# kekdata = [x / tickrate for x in range(len(datas[mat_data['titles'][0]]))][188::100]
# counter = len(kekdata)
# for kek in kekdata:
# 	plt.axvline(x=kek, linestyle="--", color="gray")
# #print("counter = ", counter)
# for j in range(counter - 1):
# 	for i in range(start, start + offset, 1):
# 		sliced_values.append(values[i])
# 		datas_times.append(i * 0.00025)
# 	for c in range (1, len(sliced_values) - 1):
# 		if (sliced_values[c - 1] < sliced_values[c] > sliced_values[c + 1]):
# 			datas_max.append(sliced_values[c])
# 			datas_max_time.append(datas_times[c])
# 		if (sliced_values[c - 1] > sliced_values[c] < sliced_values[c + 1]):
# 			datas_min.append(sliced_values[c])
# 			datas_min_time.append(datas_times[c])
# 	start += 100
# 	sliced_values.clear()
# print(len(datas_max), "max = ", datas_max)
# print(len(datas_max_time), "max_times = ", datas_max_time)
# print(len(datas_min), "min = ", datas_min, )
# print(len(datas_min_time), "min_times = ", datas_min_time)
#
#
