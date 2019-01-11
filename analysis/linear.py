import numpy as np
import pylab as plt
from sklearn.linear_model import LinearRegression
from analysis.hystograms_latency_amplitude import sim_process, find_ees_indexes, calc_amplitudes
from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.functions import read_nest_data, read_neuron_data, read_bio_data, normalization, find_latencies

bio_step = 0.25
sim_step = 0.025
delta_step = 0.3

k_bio_volt = 0
k_bio_stim = 1
k_mean = 0
k_lat = 1
k_amp = 2

def plot_linnear(bio_pack, sim_pack):
	sim_mean_volt = sim_pack[k_mean]
	sim_latencies = list(zip(*sim_pack[k_lat]))
	bio_volt = bio_pack[k_mean]
	bio_latencies = bio_pack[k_lat]

	# ToDo возвращать latency относительно EES ответа или стимуляции??

	slice_indexes = range(len(sim_latencies))

	for slice_index in slice_indexes:
		start = int(slice_index * 25 / 0.25)
		end = int((slice_index + 1) * 25 / 0.25)
		sliced_data = bio_volt[start:end]
		offset = slice_index * delta_step
		plt.plot([t * bio_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#FF7254')

	x = np.array([lat / bio_step for lat in bio_latencies])
	y_data = [slice_index * delta_step + bio_volt[int(bio_step * latency)] for slice_index, latency in enumerate(bio_latencies)]
	y = np.array(y_data)

	model = LinearRegression(fit_intercept=True)

	model.fit(x[:, np.newaxis], y)

	xfit = np.linspace(0, 10, 2)
	yfit = model.predict(xfit[:, np.newaxis])

	plt.scatter([t * bio_step for t in x], y, color='#FF4B25')
	plt.plot([t * bio_step for t in xfit], yfit,
	         color='#CA2D0C', linestyle='--', linewidth=3, label='BIO')
	plt.xlim(0, 25)
	plt.show()

	mins_by_slice = []
	index_by_slice = []

	data, indexes = read_bio_data('/home/alex/Downloads/21cms.txt')
	sliced_data = {k: [] for k in range(6)}
	for index, d in enumerate(data[:600]):
		#print(index, index // (25 / bio_step))
		sliced_data[index // (25 / bio_step)].append(d)

	for slice_index, data_loc in enumerate(sliced_data.values()):
		offset = slice_index * delta_step
		kost = 48
		min_data, min_indexes = find_mins(data_loc[kost:], 20)
		norm_data = normalization_zero_one(data_loc)
		if slice_index in [3, 5]:
			m_index = min_data.index(min_data[2])
		else:
			m_index = min_data.index(min(min_data))
		mins_by_slice.append(min_indexes[m_index] + kost)
		index_by_slice.append(offset + norm_data[min_indexes[m_index] + kost])

		plt.plot([t * bio_step for t in range(len(data_loc))],
		         [offset + d for d in norm_data],
		         color='#FF7254')

	x = np.array(mins_by_slice)
	y = np.array(index_by_slice)

	model = LinearRegression(fit_intercept=True)

	model.fit(x[:, np.newaxis], y)

	xfit = np.linspace(40, 100, 2)
	yfit = model.predict(xfit[:, np.newaxis])

	plt.yticks([])
	plt.ylabel("Slices")
	plt.xlabel("Time, ms")

	plt.scatter([t * bio_step for t in x], y, color='#FF4B25')
	plt.plot([t * bio_step for t in xfit], yfit,
	         color='#CA2D0C', linestyle='--', linewidth=3, label='BIO')
	plt.legend()
	plt.show()


def process_bio_data(data, slice_numbers):
	bio_stim_indexes = data[k_bio_stim][:slice_numbers + 1]
	bio_voltages = data[k_bio_volt][:bio_stim_indexes[-1]]
	bio_datas = calc_max_min(bio_stim_indexes, bio_voltages, bio_step)
	bio_ees_indexes = find_ees_indexes(bio_stim_indexes[:-1], bio_datas)
	norm_bio_voltages = normalization(bio_voltages, zero_relative=True)
	bio_datas = calc_max_min(bio_ees_indexes, norm_bio_voltages, bio_step, remove_micropeaks=True)
	print(bio_datas)
	raise Exception
	bio_lat = find_latencies(bio_datas, bio_step, norm_to_ms=True)
	bio_amp = calc_amplitudes(bio_datas, bio_lat)

	return bio_lat, bio_amp


def run():
	# get data
	bio_voltages = read_bio_data('/home/alex/bio_21cms.txt')
	nest_tests = read_nest_data('/home/alex/nest_21cms.hdf5')
	neuron_tests = read_neuron_data('/home/alex/neuron_21cms.hdf5')

	slice_numbers = int(len(neuron_tests[0]) * sim_step // 25)

	bio_lat, bio_amp = process_bio_data(bio_voltages, slice_numbers)
	bio_pack = (normalization(bio_voltages[0], zero_relative=True), bio_lat)

	# the main loop of simulations data
	for sim_datas in [nest_tests, neuron_tests]:
		sim_lat = []
		sim_amp = []
		# collect amplitudes and latencies per test data
		for test_data in sim_datas:
			lat, amp = sim_process(test_data)
			sim_lat.append(lat)
			sim_amp.append(amp)

		sim_mean_volt = list(map(lambda voltages: np.mean(voltages), zip(*sim_datas)))
		sim_pack = (normalization(sim_mean_volt, zero_relative=True), sim_lat)

		plot_linnear(bio_pack, sim_pack)


if __name__ == "__main__":
	run()
