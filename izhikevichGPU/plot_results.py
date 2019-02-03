import sys
import numpy as np
import pylab as plt 
import numpy.ma as ma
import logging as log

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotting')

nrns_id_start = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1769, 1938, 1978, 2018, 2058, 2098, 2138, 2178, 2218, 2258]

groups_name = ["C1", "C2", "C3", "C4", "C5", "D1_1", "D1_2", "D1_3", "D1_4", "D2_1", "D2_2", "D2_3", "D2_4", "D3_1", "D3_2", "D3_3", "D3_4", "D4_1", "D4_2", "D4_3", "D4_4", "D5_1", "D5_2", "D5_3", "D5_4", "G1_1", "G1_2", "G1_3", "G2_1", "G2_2", "G2_3", "G3_1", "G3_2", "G3_3", "G4_1", "G4_2", "G4_3", "G5_1", "G5_2", "G5_3", "IP_E", "MP_E", "EES", "inh_group3", "inh_group4", "inh_group5", "ees_group1", "ees_group2", "ees_group3", "ees_group4"]


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

		plt.figure(figsize=(10, 5))
		plt.suptitle(title)

		ax1 = plt.subplot(211)
		plt.plot([x * step for x in range(len(voltages))], voltages, color='b', label='voltages')
		plt.plot(spikes, [0] * len(spikes), '.', color='r')

		for slice_index in range(6):
			plt.axvline(x=slice_index * 25, color='grey', linestyle='--')
		plt.legend()
		plt.xlim(0, len(voltages) * step)

		plt.subplot(212, sharex=ax1)
		t = [x * step for x in range(len(currents))]

		plt.plot(t, currents, color='r')
		plt.plot(t, [0] * len(currents), color='grey')
		
		for slice_index in range(6):
			plt.axvline(x=slice_index * 25, color='grey', linestyle='--')

		plt.legend()
		plt.xlim(0, len(voltages) * step)

		filename = "{}.png".format(title)

		plt.savefig("{}/{}".format(save_to, filename), format="png")

		plt.show()
		plt.close()

		logger.info(title)


def run():
	step = 0.1
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

