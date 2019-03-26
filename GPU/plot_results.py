import sys
import pylab as plt
import logging as log

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotting')

nrns_id_start = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 
                 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 1016, 
                 1212, 1381, 1550, 1610, 1670, 1690, 1710, 1730, 1750, 1770, 1790, 1810, 1830, 1850, 1870, 1890, 1910, 1930]

groups_name = ["D1_3", "D2_3", "D4_3", "G3_1", "G3_2", "CV1", "CV2", "CV3", "CV4", "CV5", "EES", "D1_1", "D1_2", "D1_4", 
               "D2_1", "D2_2", "D2_4", "D3_1", "D3_2", "D3_3", "D3_4", "D4_1", "D4_2", "D4_4", "D5_1", "D5_2", "D5_3", 
               "D5_4", "G1_1", "G1_2", "G1_3", "G2_1", "G2_2", "G2_3", "G3_3", "G4_1", "G4_2", "G4_3", "G5_1", "G5_2", 
               "G5_3", "IP_E", "IP_F", "MP_E", "MP_F", "Ia_Extensor", "Ia_Flexor", "inh_group3", "inh_group4", "inh_group5", 
               "ees_group1", "ees_group2", "ees_group3", "ees_group4", "R_E", "R_F", "Ia_E", "Ia_F", "Ib_E", "Ib_F"]

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


def draw_slice_borders(sim_time):
	a = [125]
	for i in range(1, 20):
		a.append(a[i - 1] + (125 if i % 2 == 0 else 150) )
	# for i in range(0, sim_time, 25):
	for i in a:
		plt.axvline(x=i, linewidth=3, color='k')
	for i in range(0, sim_time, 25):
		plt.axvline(x=i, linewidth=1, color='k')


def plot(global_volts, global_currents, global_spikes, step, save_to):
	for volt_per_nrn, curr_per_nrn, spikes_per_nrn in zip(global_volts.items(),
		                                                  global_currents.items(),
		                                                  global_spikes.items()):
		title = volt_per_nrn[0]
		voltages = volt_per_nrn[1]
		currents = curr_per_nrn[1]
		spikes = spikes_per_nrn[1]

		sim_time = int(len(voltages) * step)
		slices_number = int(len(voltages) * step / 25)

		if title == "MP_E" or title == "MP_F":
			plt.figure(figsize=(10, 5))
			plt.suptitle("Sliced {}".format(title))
			V_rest = 72
			for slice_index in range(0 if title is "MP_F" else 5, 5 if title is "MP_F" else slices_number):
				chunk_start = int(slice_index * 25 / step)
				chunk_end = int((slice_index + 1) * 25 / step)
				Y = [-v - V_rest for v in voltages[chunk_start:chunk_end]]
				shifted_y = [y + 30 * slice_index for y in Y]
				X = [x * step for x in range(len(Y))]
				plt.plot(X, shifted_y, linewidth=0.7)
			plt.xlim(0, 25)
			plt.yticks([])

			with open("/home/alex/{}.dat".format(title), "w") as file:
				for volt in voltages:
					file.write("{} ".format(volt))

			filename = "sliced_{}.png".format(title)
			plt.savefig("{}/{}".format(save_to, filename), format="png")
			plt.close()

		plt.figure(figsize=(10, 5))
		plt.suptitle(title)

		# 1
		ax1 = plt.subplot(311)

		draw_slice_borders(sim_time)

		plt.plot([x * step for x in range(len(voltages))], voltages, color='b')
		plt.plot(spikes, [0] * len(spikes), '.', color='r', alpha=0.7)

		plt.xlim(0, sim_time)
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.ylabel("Voltage, mV")

		plt.grid()

		# 2
		plt.subplot(312, sharex=ax1)
		t = [x * step for x in range(len(currents))]

		draw_slice_borders(sim_time)

		plt.plot(t, currents, color='r')
		plt.plot(t, [0] * len(currents), color='grey')

		plt.ylabel("Currents, pA")

		plt.xlim(0, sim_time)
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.grid()

		# 3
		plt.subplot(313, sharex=ax1)

		draw_slice_borders(sim_time)

		plt.hist(spikes, bins=range(sim_time))
		plt.xlim(0, sim_time)
		plt.grid()

		plt.ylabel("Spikes, n")
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25),
		           ["{}\n{}".format(global_time - 25 * (global_time // 25), global_time)
		            for global_time in range(0, sim_time + 1, 5 if sim_time <= 275 else 25)],
		           fontsize=8)

		plt.subplots_adjust(hspace=0.05)

		filename = "{}.png".format(title)

		plt.savefig("{}/{}".format(save_to, filename), format="png")
		plt.close()

		logger.info(title)


def scientific_plot(neurons_volt):
	import pyqtgraph



def run():
	step = 0.25
	form_in_group = True
	path = sys.argv[1]

	filepath_volt = path + "/volt_0.dat"
	filepath_curr = path + "/curr_0.dat"
	filepath_spike = path + "/spikes_0.dat"

	neurons_volt = read_data(filepath_volt)
	neurons_curr = read_data(filepath_curr)
	neurons_spike = read_data(filepath_spike)

	neurons_volt = process_data(neurons_volt, form_in_group=form_in_group)
	neurons_curr = process_data(neurons_curr, form_in_group=form_in_group)
	neurons_spike = process_data_spikes(neurons_spike, form_in_group=form_in_group)

	plot(neurons_volt, neurons_curr, neurons_spike, step=step, save_to=path + "/results/")

	# scientific_plot(neurons_volt)

if __name__ == "__main__":
	run()

