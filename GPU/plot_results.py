import os
import sys
import pylab as plt
import logging as log

log.basicConfig(format='%(name)s::%(funcName)s %(message)s', level=log.INFO)
logger = log.getLogger('Plotting')

def read_data(path):
	names = []
	voltage = []
	g_exc = []
	g_inh = []
	spikes = []

	for filename in [f for f in sorted(os.listdir(path)) if f.endswith(".dat")]:
		with open(f"{path}/{filename}") as file:
			names.append(filename.replace(".dat", ""))
			voltage.append(list(map(float, file.readline().split())))
			g_exc.append(list(map(float, file.readline().split())))
			g_inh.append(list(map(float, file.readline().split())))
			spikes.append(list(map(float, file.readline().split())))
	logger.info(f"done : {path}")

	return names, voltage, g_exc, g_inh, spikes


def draw_slice_borders(sim_time):
	a = [125]
	for i in range(1, 20):
		a.append(a[i - 1] + (125 if i % 2 == 0 else 150) )
	# for i in range(0, sim_time, 25):
	for i in a:
		plt.axvline(x=i, linewidth=3, color='k')
	for i in range(0, sim_time, 25):
		plt.axvline(x=i, linewidth=1, color='k')


def plot(names, voltages, g_exc, g_inh, spikes, step, save_to):
	for name, voltage, g_e, g_i, s in zip(names, voltages, g_exc, g_inh, spikes):
		sim_time = int(len(voltage) * step)
		slices_number = int(len(voltage) * step / 25)
		shared_x = list(map(lambda x: x * step, range(len(voltage))))

		if name == "MP_E" or name == "MP_F":
			plt.figure(figsize=(16, 9))
			plt.suptitle(f"Sliced {name}")
			V_rest = 72

			for slice_index in range(0 if name is "MP_F" else 5, 5 if name is "MP_F" else slices_number):
				chunk_start = int(slice_index * 25 / step)
				chunk_end = int((slice_index + 1) * 25 / step)
				Y = [-v - V_rest for v in voltage[chunk_start:chunk_end]]
				shifted_y = [y + 30 * slice_index for y in Y]
				X = [x * step for x in range(len(Y))]
				plt.plot(X, shifted_y, linewidth=0.7)
			plt.xlim(0, 25)
			plt.yticks([])
			plt.show()
			plt.close()

		plt.figure(figsize=(16, 9))
		plt.suptitle(name)

		# 1
		ax1 = plt.subplot(311)
		draw_slice_borders(sim_time)

		plt.plot(shared_x, voltage, color='b')
		plt.plot(s, [0] * len(s), '.', color='r', alpha=0.7)

		plt.xlim(0, sim_time)
		# a small hotfix to hide x lables but save x ticks -- set them as white color
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.ylabel("Voltage, mV")

		plt.grid()

		# 2
		plt.subplot(312, sharex=ax1)
		draw_slice_borders(sim_time)

		plt.plot(shared_x, g_e, color='r')
		plt.plot(shared_x, g_i, color='b')
		plt.plot(shared_x, [0] * len(voltage), color='grey')

		plt.ylabel("Currents, pA")

		plt.xlim(0, sim_time)
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.grid()

		# 3
		plt.subplot(313, sharex=ax1)
		draw_slice_borders(sim_time)

		plt.hist(s, bins=range(sim_time))   # bin is equal to 1ms
		plt.xlim(0, sim_time)
		plt.grid()

		plt.ylabel("Spikes, n")
		plt.ylim(bottom=0)
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25),
		           ["{}\n{}".format(global_time - 25 * (global_time // 25), global_time)
		            for global_time in range(0, sim_time + 1, 5 if sim_time <= 275 else 25)],
		           fontsize=8)

		plt.subplots_adjust(hspace=0.08)

		plt.savefig(f"{save_to}/{name}.png", format="png", dpi=200)
		plt.close()

		logger.info(name)


def scientific_plot(neurons_volt):
	import pyqtgraph
	pass


def run():
	step = 0.025
	if len(sys.argv) > 1:
		path = sys.argv[1]
	else:
		path = "/home/alex/GitHub/memristive-spinal-cord/GPU/matrix_solution/dat/"

	plot(*read_data(path), step=step, save_to=path + "/results/")

	# scientific_plot(neurons_volt)

if __name__ == "__main__":
	run()

