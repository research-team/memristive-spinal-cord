import pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from functions import read_bio_data, read_NEST_data, normalization_zero_one, find_mins


def processing():
	"""
	ToDo: write an annotation
	Returns:

	"""
	datas = read_NEST_data('/home/alex/Downloads/nest_21cms.hdf5')
	delta_step = 0.3

	sim_step = 0.025
	bio_step = 0.25

	mins_by_slice = []
	index_by_slice = []

	tests = {}

	for test_index, data in enumerate(datas.values()):
		sliced_data = {k: [] for k in range(6)}

		for index, d in enumerate(data):
			sliced_data[index // (25 / sim_step)].append(d)

		tests[test_index] = sliced_data

		for slice_index, data_loc in enumerate(sliced_data.values()):
			offset = slice_index * delta_step
			min_data, min_indexes = find_mins(data_loc[220:], 60)
			norm_data = normalization_zero_one(data_loc)
			mins_by_slice.append(min_indexes[0] + 220)
			index_by_slice.append(offset + norm_data[min_indexes[0] + 220])

	med = {k: [] for k in range(6)}

	for test_index, test_sliced in tests.items():
		for slice_index, slice_data in test_sliced.items():
			med[slice_index].append(slice_data)
	for k, v in med.items():
		med[k] = list(map(lambda x: np.mean(x), zip(*v)))
		plt.plot([t * sim_step for t in range(len(med[k]))],
		         [d + k * delta_step for d in normalization_zero_one(med[k])],
		         color='#54CBFF')
	x = np.array(mins_by_slice)
	y = np.array(index_by_slice)

	model = LinearRegression(fit_intercept=True)

	model.fit(x[:, np.newaxis], y)

	xfit = np.linspace(400, 1000, 50)
	yfit = model.predict(xfit[:, np.newaxis])

	plt.scatter([t * sim_step for t in x], y, color="#25C7FF", alpha=0.3)
	plt.plot([t * sim_step for t in xfit],
	         yfit, color='#0C88CA', linestyle='--', linewidth=3)

	plt.xlim(0, 25)


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

	plt.scatter([t * bio_step for t in x], y, color='#FF4B25')
	plt.plot([t * bio_step for t in xfit], yfit,
	         color='#CA2D0C', linestyle='--', linewidth=3)

	plt.show()


def run():
	"""
	ToDo: write an annotation
	ToDo: split logic blocks into functions
	Returns:

	"""
	processing()


if __name__ == "__main__":
	run()
