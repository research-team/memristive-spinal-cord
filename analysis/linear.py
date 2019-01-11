import numpy as np
import pylab as plt
from sklearn.linear_model import LinearRegression
from analysis.hystograms_latency_amplitude import sim_process, bio_process
from analysis.functions import read_nest_data, read_neuron_data, read_bio_data, normalization

bio_step = 0.25
sim_step = 0.025
delta_step = 0.3

k_bio_volt = 0
k_bio_stim = 1
k_mean = 0
k_lat = 1
k_amp = 2


def plot_linear(bio_pack, sim_pack):
	"""
	Args:
		bio_pack:
		sim_pack:
	"""
	sim_mean_volt = sim_pack[k_mean]
	sim_x = sim_pack[k_lat]
	sim_y = sim_pack[k_amp]

	bio_volt = bio_pack[k_mean]
	bio_x = bio_pack[k_lat]
	bio_y = bio_pack[k_amp]

	slice_indexes = range(6)

	for slice_index in slice_indexes:
		start = int(slice_index * 25 / bio_step)
		end = int((slice_index + 1) * 25 / bio_step)
		sliced_data = bio_volt[start:end]
		offset = slice_index * delta_step
		plt.plot([t * bio_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#FF7254')

		start = int(slice_index * 25 / sim_step)
		end = int((slice_index + 1) * 25 / sim_step)
		sliced_data = sim_mean_volt[start:end]
		plt.plot([t * sim_step for t in range(len(sliced_data))],
		         [offset + d for d in sliced_data],
		         color='#54CBFF')

	x = np.array(bio_x)
	y = np.array(bio_y)

	model = LinearRegression(fit_intercept=True)
	model.fit(x[:, np.newaxis], y)

	xfit = np.linspace(0, 25, 2)
	yfit = model.predict(xfit[:, np.newaxis])

	plt.scatter(x, y, color='#FF4B25', alpha=0.3)
	plt.plot(xfit, yfit, color='#CA2D0C', linestyle='--', linewidth=3, label='BIO')
	plt.xlim(0, 25)


	x = np.array(sim_x)
	y = np.array(sim_y)

	model = LinearRegression(fit_intercept=True)
	model.fit(x[:, np.newaxis], y)

	xfit = np.linspace(0, 25, 2)
	yfit = model.predict(xfit[:, np.newaxis])

	plt.xlabel("Time, ms")
	plt.ylabel("Slices")
	plt.ylim(-1, 1.8)
	plt.yticks([])

	plt.scatter(x, y, color='#25C7FF', alpha=0.3)
	plt.plot(xfit, yfit, color='#0C88CA', linestyle='--', linewidth=3, label='NEST')
	plt.legend()
	plt.show()


def run():
	# get data
	bio_voltages = read_bio_data('/home/alex/bio_21cms.txt')
	nest_tests = read_nest_data('/home/alex/nest_21cms.hdf5')
	neuron_tests = read_neuron_data('/home/alex/neuron_21cms.hdf5')

	slice_numbers = int(len(neuron_tests[0]) * sim_step // 25)

	bio_lat, bio_amp = bio_process(bio_voltages, slice_numbers)

	bio_voltages = normalization(bio_voltages[0], zero_relative=True)
	bio_x = bio_lat
	bio_y = [index * delta_step + bio_voltages[int((bio_x[index] + 25 * index) / bio_step)]
	         for index in range(slice_numbers)]

	bio_pack = (bio_voltages, bio_x, bio_y)

	# the main loop of simulations data
	for sim_datas in [nest_tests, neuron_tests]:
		sim_x = []
		sim_y = []
		# collect amplitudes and latencies per test data
		for test_data in sim_datas:
			lat, amp = sim_process(test_data)
			sim_x.append(lat)

			tmp = []
			test_data = normalization(test_data, zero_relative=True)
			for slice_index in range(len(lat)):
				tmp.append(slice_index * delta_step + test_data[int(lat[slice_index] / sim_step + slice_index * 25 / sim_step)])
			sim_y.append(tmp)
		sim_x = sum(sim_x, [])
		sim_y = sum(sim_y, [])
		sim_mean_volt = normalization(list(map(lambda voltages: np.mean(voltages), zip(*sim_datas))), zero_relative=True)
		sim_pack = (sim_mean_volt, sim_x, sim_y)

		plot_linear(bio_pack, sim_pack)


if __name__ == "__main__":
	run()
