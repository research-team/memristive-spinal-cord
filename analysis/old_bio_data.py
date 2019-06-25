from analysis.patterns_in_bio_data import bio_data_runs
from matplotlib import pylab as plt
from analysis.real_data_slices import read_data
from analysis.functions import read_bio_data

bio_step = 0.25


def plot_mat_data():
	mat_data = read_data('../bio-data/SCI_Rat-1_11-22-2016_40Hz_RTA_one step.mat')
	voltages = mat_data['data'][0][:int(len(mat_data['data'][0]) / 2)]
	stims = mat_data['data'][0][int(len(mat_data['data'][0]) / 2):]
	# print(mat_data['titles'])
	# print(voltages)

	# print("len(voltages) = ", len(voltages))
	# plt.plot(voltages)
	# plt.show()

	# plt.plot(stims)
	# plt.show()

	volt_data = []
	stim_data = []
	slices_begin_time = []

	title_stim = 'Stim'
	title_rmg = 'RMG'
	title_rta = 'RTA'

	for index, data_title in enumerate(mat_data['titles']):
		print("index = ", index)
		print("data_title = ", data_title)
		data_start = int(mat_data['datastart'][index]) - 1
		# print("data_start = ", data_start)
		data_end = int(mat_data['dataend'][index])
		# print("data_end = ", data_end)
		float_data = [round(float(x), 3) for x in mat_data['data'][0][data_start:data_end]]
		# print("float_data = ", float_data)
		# print("data_title = ", data_title)
		if title_rmg in data_title:
			volt_data = float_data
		if title_rta in data_title:
			volt_data = float_data
		if title_stim in data_title:
			stim_data = float_data
		print("---")
	ms_pause = 0

	print("len(stim_data) = ", len(stim_data))
	for index in range(1, len(stim_data) - 1):
		# print("for")
		if stim_data[index - 1] < stim_data[index] > stim_data[index + 1] and ms_pause <= 0 and stim_data[index] > 0.5:
			# print("in")
			slices_begin_time.append(index)  # * real_data_step  # division by 4 gives us the normal 1 ms step size
			ms_pause = int(3 / bio_step)
		ms_pause -= 1
				# raw_stim = [i for i, d in enumerate(float_data) if d > 0.5]
				# raw_stim = list(map(lambda x: x - raw_stim[0], raw_stim))
				# o = 0
				# i = 0
				#
				# for d in raw_stim:
				# 	if len(slices_begin_time) != 0 and d - 10 < slices_begin_time[i - 1] < d + 10:
				# 		continue
				# 	if o - 10 < d < o + 10:
				# 		slices_begin_time.append(d)
				# 		o += 100
				# 		i += 1
	print("slices_begin_time = ", slices_begin_time)
	volt_data = volt_data[slices_begin_time[0]:slices_begin_time[-1]]

	# move times to the begin (start from 0 ms)
	slices_begin_time = [t - slices_begin_time[0] for t in slices_begin_time]

	print("volt_data = ", volt_data)
	print("slices_begin_time = ", slices_begin_time)

	offset = 0
	volt_slices = []
	for i in range(int(len(volt_data) / 100)):  # 100 for 40Hz, 200 for 20Hz
		volt_slices_tmp = []
		for j in range(offset, offset + 100):
			volt_slices_tmp.append(volt_data[j])
		volt_slices.append(volt_slices_tmp)
		offset += 100

	# for slice in volt_slices:
		# print("slice = ", len(slice), slice)

	yticks = []
	for index, sl in enumerate(volt_slices):
		offset = index * 2
		times = [time * bio_step for time in range(len(volt_slices[0]))]
		plt.plot(times, [s + offset for s in sl])
		yticks.append(sl[0] + offset)
	plt.yticks(yticks, range(1, len(volt_slices) + 1))
	plt.xlim(0, 25)
	plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	plt.show()

	print("volt_slices = ", len(volt_slices))
	plt.plot(volt_data)
	print("len(volt_data) = ", len(volt_data))
	data = bio_data_runs()
	print(len(data), data)
	# plt.plot(data)
	plt.show()


def plot_txt_data():
	data = read_bio_data('../bio-data/4_Rat-16_5-09-2017_RMG_one_step_T.txt')
	voltages = data[0]
	stims = data[1]
	print("stims = ", stims)
	offset = 0
	volt_slices = []
	for i in range(int(len(voltages) / 100)):  # 100 for 40Hz, 200 for 20Hz
		volt_slices_tmp = []
		for j in range(offset, offset + 100):
			volt_slices_tmp.append(voltages[j])
		volt_slices.append(volt_slices_tmp)
		offset += 100

	yticks = []
	for index, sl in enumerate(volt_slices):
		offset = index * 2
		times = [time * bio_step for time in range(len(volt_slices[0]))]
		plt.plot(times, [s + offset for s in sl])
		yticks.append(sl[0] + offset)
	plt.yticks(yticks, range(1, len(volt_slices) + 1))
	plt.xlim(0, 25)
	plt.grid(which='major', axis='x', linestyle='--', linewidth=0.5)
	plt.show()

def main():
	plot_txt_data()


if __name__ == '__main__':
    main()