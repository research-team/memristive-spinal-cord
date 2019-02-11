import sys
import pylab as plt
import logging as log

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotting')

nrns_id_start = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360,
                 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720,
                 740, 760, 780, 800, 996, 1192, 1361, 1530, 1550, 1570, 1766, 1786, 1806, 1826, 1846, 1866,
                 1886, 1906, 1926, 1946, 1966, 1986, 2006, 2026, 2195, 2364, 2384, 2404]

groups_name = ["C1", "C2", "C3", "C4", "C5", "D1_1", "D1_2", "D1_3", "D1_4", "D2_1", "D2_2", "D2_3", "D2_4",
               "D3_1", "D3_2", "D3_3", "D3_4", "D4_1", "D4_2", "D4_3", "D4_4", "D5_1", "D5_2", "D5_3", "D5_4",
               "G1_1", "G1_2", "G1_3", "G2_1", "G2_2", "G2_3", "G3_1", "G3_2", "G3_3", "G4_1", "G4_2", "G4_3",
               "G5_1", "G5_2", "G5_3", "IP_E", "IP_F", "MP_E", "MP_F", "EES", "I5", "Ia", "inh_group3",
               "inh_group4", "inh_group5", "ees_group1", "ees_group2", "ees_group3", "ees_group4", "R_E", "R_F",
               "Ia_E", "Ia_F", "Ib_E", "Ib_F", "Extensor", "Flexor", "C_0", "C_1"]


def read_data(path):
	data_container = {}
	with open(path) as file:
		for line in file:
			gid, *data = line.split()
			data_container[int(gid)] = [float(d) for d in data]
	logger.info("done : {}".format(path))
	return data_container


def process_data(data, form_in_group):
	logger.info("Start processing...")
	if form_in_group:
		combined_data = {}

		for group_index in range(len(nrns_id_start) - 1):
			group = groups_name[group_index]
			neurons_number = nrns_id_start[group_index + 1] - nrns_id_start[group_index]
			first_nrn_in_group = nrns_id_start[group_index]

			if group not in combined_data.keys():
				combined_data[group] = [v / neurons_number for v in data[first_nrn_in_group]]

			for nrn_id in range(first_nrn_in_group + 1, first_nrn_in_group + neurons_number):
				combined_data[group] = [a + b / neurons_number for a, b in zip(combined_data[group], data[nrn_id])]
		return combined_data
	return data


def process_data_spikes(data, form_in_group):
	logger.info("Start processing...")
	if form_in_group:
		combined_data = {}

		for group_index in range(len(nrns_id_start) - 1):
			group = groups_name[group_index]
			neurons_number = nrns_id_start[group_index + 1] - nrns_id_start[group_index]
			first_nrn_in_group = nrns_id_start[group_index]

			if group not in combined_data.keys():
				combined_data[group] = data[first_nrn_in_group]

			for nrn_id in range(first_nrn_in_group + 1, first_nrn_in_group + neurons_number):
				combined_data[group] += data[nrn_id]
		return combined_data
	return data


def plot(global_volts, global_currents, global_spikes, step, save_to):
	for volt_per_nrn, curr_per_nrn, spikes_per_nrn in zip(global_volts.items(),
		                                                  global_currents.items(),
		                                                  global_spikes.items()):
		title = volt_per_nrn[0]
		voltages = volt_per_nrn[1]
		currents = curr_per_nrn[1]
		spikes = spikes_per_nrn[1]

		if title == "MP_E":
			plt.figure(figsize=(10, 5))
			plt.suptitle("Sliced {}".format(title))
			V_rest = 72
			for slice_index in range(6):
				chunk_start = int(slice_index * 25 / step)
				chunk_end = int((slice_index + 1) * 25 / step)
				Y = [-v - V_rest for v in voltages[chunk_start:chunk_end]]
				shifted_y = [y + 10 * slice_index for y in Y]
				X = [x * step for x in range(len(Y))]
				plt.plot(X, shifted_y, linewidth=0.7)
			plt.xlim(0, 25)
			plt.yticks([])

			filename = "sliced_{}.png".format(title)
			plt.savefig("{}/{}".format(save_to, filename), format="png")
			plt.close()

		plt.figure(figsize=(10, 5))
		plt.suptitle(title)

		ax1 = plt.subplot(211)
		plt.plot([x * step for x in range(len(voltages))], voltages, color='b', label='voltages')
		plt.plot(spikes, [0] * len(spikes), '.', color='r', alpha=0.7)

		for slice_index in range(6):
			plt.axvline(x=slice_index * 25, color='grey', linestyle='--')
		plt.legend()
		plt.xlim(0, len(voltages) * step)

		plt.subplot(212, sharex=ax1)
		t = [x * step for x in range(len(currents))]

		plt.plot(t, currents, color='r')
		plt.plot(t, [0] * len(currents), color='grey')

		filename = "{}.png".format(title)

		for slice_index in range(6):
			plt.axvline(x=slice_index * 25, color='grey', linestyle='--')

		plt.legend()
		plt.xlim(0, len(voltages) * step)

		plt.savefig("{}/{}".format(save_to, filename), format="png")
		plt.close()

		logger.info(title)


def run():
	step = 0.25
	form_in_group = True
	path = sys.argv[1]

	filepath_volt = path + "/volt.dat"
	filepath_curr = path + "/curr.dat"
	filepath_spike = path + "/spikes.dat"

	neurons_volt = read_data(filepath_volt)
	neurons_curr = read_data(filepath_curr)
	neurons_spike = read_data(filepath_spike)

	neurons_volt = process_data(neurons_volt, form_in_group=form_in_group)
	neurons_curr = process_data(neurons_curr, form_in_group=form_in_group)
	neurons_spike = process_data_spikes(neurons_spike, form_in_group=form_in_group)

	plot(neurons_volt, neurons_curr, neurons_spike, step=step, save_to=path + "/results/")


if __name__ == "__main__":
	run()

