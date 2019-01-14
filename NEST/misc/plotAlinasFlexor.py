import os
import pylab
import numpy as np

cores = []
sim_step = 0.025
sim_time = 125
neurons_data = []
global_spike_data = []
EES_neurons_data = []


def create_group(name, begin_index):
	cores.append( (name, begin_index * 2) )


def init_groups():
	create_group("D1_1", 0)
	create_group("D1_2", 50)
	create_group("D1_3", 150)
	create_group("D1_4", 100)

	create_group("D2_1", 10)
	create_group("D2_2", 60)
	create_group("D2_3", 160)
	create_group("D2_4", 110)

	create_group("D3_1", 20)
	create_group("D3_2", 70)
	create_group("D3_3", 170)
	create_group("D3_4", 120)

	create_group("D4_1", 30)
	create_group("D4_2", 80)
	create_group("D4_3", 180)
	create_group("D4_4", 130)

	create_group("D5_1", 40)
	create_group("D5_2", 90)
	create_group("D5_3", 190)
	create_group("D5_4", 140)

	create_group("G1_1", 200)
	create_group("G1_2", 250)
	create_group("G1_3", 300)

	create_group("G2_1", 210)
	create_group("G2_2", 260)
	create_group("G2_3", 310)

	create_group("G3_1", 220)
	create_group("G3_2", 270)
	create_group("G3_3", 320)

	create_group("G4_1", 230)
	create_group("G4_2", 280)
	create_group("G4_3", 330)

	create_group("G5_1", 240)
	create_group("G5_2", 290)
	create_group("G5_3", 340)

	create_group("E1", 350)
	create_group("E2", 360)
	create_group("E3", 370)
	create_group("E4", 380)


def read(path):
	global EES_neurons_data
	global neurons_data
	global global_spike_data

	for neuron_id in range(390):
		for index in [0, 1]:
			name = "vIn{0}r{1}s25v0".format(neuron_id, index)
			print(name)
			tmp_voltages = []
			with open(os.path.join(path, name)) as file:
				for line in file:
					tmp_voltages.append(float(line))
				del tmp_voltages[0]

			neurons_data.append(tmp_voltages)
			# find spikes
			prev_time = -999
			tmp_spikes = []
			for j in range(1, len(tmp_voltages) - 1):
				if tmp_voltages[j - 1] < tmp_voltages[j] > tmp_voltages[j + 1]:
					if tmp_voltages[j] >= 0 and abs(prev_time - j) > 40:
						tmp_spikes.append(j)
						prev_time = j
			global_spike_data.append(tmp_spikes)


	for neuron_id in range(84):
		for index in [0, 1]:
			name = "vMN{0}r{1}s25v0".format(neuron_id, index)
			tmp_list = []
			with open(os.path.join('/home/alex/Documents/AlinasResults/vMN/', name)) as file:
				for line in file:
					tmp_list.append(float(line) * 10**10)
				del tmp_list[0]
			EES_neurons_data.append(tmp_list)
			print(name)
	EES_neurons_data = list(map(lambda x: np.mean(x), zip(*EES_neurons_data)))


def plot():
	# plot NODES
	for core in cores:
		name = core[0]
		begin_index = core[1]
		end_index = begin_index + 20
		print(name, begin_index, end_index)

		pylab.figure(figsize=(10, 5))
		pylab.plot([x * sim_step for x in range(len(neurons_data[begin_index]))],
		           list(map(lambda x: np.mean(x), zip(*[neurons_data[key] for key in range(begin_index, end_index)]))))
		spikes = [global_spike_data[key] for key in range(begin_index, end_index)]
		spikes = [j * sim_step for sub in spikes for j in sub]
		pylab.plot(spikes, [0]*len(spikes), '.', color='r', markersize=1)
		# plot the slice border
		for i in np.arange(0, sim_time, 25):
			pylab.axvline(x=i, linewidth=1, color='k')
		pylab.xlabel('Time [ms]')
		pylab.ylabel('Currents [pA]')
		pylab.xlim(0, sim_time)
		pylab.xticks(range(0, sim_time + 1, 5),
		             ["{}\n{}".format(global_time - 25 * (global_time // 25), global_time)
		              for global_time in range(0, sim_time + 1, 5)],
		             fontsize=8)
		pylab.yticks(fontsize=8)
		pylab.grid()

		pylab.savefig("/home/alex/GitHub/memristive-spinal-cord/NEST/misc/{}.png".format(name), dpi=250, format="png")
		pylab.close()

	# plot MN
	pylab.figure(figsize=(10, 5))
	# plot the slice border
	for index_begin in range(5):
		offset = index_begin * 15
		index_begin *= 1000
		volt = [voltage + offset for voltage in EES_neurons_data[index_begin:index_begin+1000]]
		times = [time * sim_step for time in range(len(volt))]
		# plot mean with shadows
		pylab.plot(times, volt, linewidth=0.8, color='k')

	pylab.xlabel('Time [ms]')
	pylab.xlim(0, 25)
	pylab.xticks(range(26), [i if i % 5 == 0 else "" for i in range(26)], fontsize=8)
	pylab.yticks(fontsize=8)
	pylab.grid()

	pylab.savefig("/home/alex/GitHub/memristive-spinal-cord/NEST/misc/Slices.png", dpi=250, format="png")
	pylab.close()

if __name__ == "__main__":
	init_groups()
	read("/home/alex/Documents/AlinasResults/vIn")
	plot()
