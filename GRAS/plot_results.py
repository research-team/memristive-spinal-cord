import os
import sys
import numpy as np
import logging as log
import matplotlib.pyplot as plt

log.basicConfig(format='[%(funcName)s]: %(message)s', level=log.INFO)
logger = log.getLogger()
slice_len = 25

def read_data(path):
	"""
	ToDo
	Args:
		path (str):
	Returns:
		names (list of str):
		voltage (list of np.array):
		g_exc (list of np.array):
		g_inh (list of np.array):
		spikes (list of np.array):
	"""
	names, voltage, g_exc, g_inh, spikes = [], [], [], [], []
	filenames = (f for f in os.listdir(path) if f.endswith(".dat"))
	for filename in filenames:
		with open(f"{path}/{filename}") as file:
			names.append(filename.replace(".dat", ""))
			voltage.append(np.array(file.readline().split(), dtype=np.float))
			g_exc.append(np.array(file.readline().split(), dtype=np.float))
			g_inh.append(np.array(file.readline().split(), dtype=np.float))
			spikes.append(np.array(file.readline().split(), dtype=np.float))
	logger.info(f"done: {path}")
	return names, voltage, g_exc, g_inh, spikes


def draw_slice_borders(sim_time, skin_stim_time):
	for i in range(0, sim_time, skin_stim_time):
		plt.axvline(x=i, linewidth=1, color='k')


def plot(skin_stim_time, names, voltages, g_exc, g_inh, spikes, step, save_to):
	"""
	ToDo
	Args:
		skin_stim_time:
		names:
		voltages:
		g_exc:
		g_inh:
		spikes:
		step:
		save_to:
	"""
	for name, voltage, g_e, g_i, sp in zip(names, voltages, g_exc, g_inh, spikes):
		# fixme: remove the artefact from the first derivative
		if any(n in name for n in ['muscle_E', 'muscle_F']):
			voltage[0], g_e[0], g_i[0] = 0, 0, 0
		# simulation time based on the data size
		sim_time = int(len(voltage) * step)
		slices_number = int(len(voltage) * step / slice_len)
		shared_x = np.arange(len(voltage)) * step
		# slice plot for muscle and motoneurons
		if any(n in name for n in ["mns_E", 'mns_F', 'muscle_E', 'muscle_F']):
			mxm = (voltage.max() - voltage.min()) / 5
			plt.figure(figsize=(16, 9))
			plt.suptitle(f"Sliced {name}")
			for slice_index in range(slices_number):
				if slice_index == 0:
					continue
				chunk_start = int(slice_index * slice_len / step)
				chunk_end = int((slice_index + 1) * slice_len / step)
				Y = np.array(voltage[chunk_start:chunk_end])
				shifted_y = Y + mxm * slice_index
				X = np.arange(len(Y)) * step
				plt.plot(X, shifted_y, linewidth=0.7, color='r')
			plt.xlim(0, slice_len)
			plt.yticks([])
			plt.savefig(f"{save_to}/sliced_{name}.png", format="png", dpi=200)
			plt.close()

		plt.figure(figsize=(16, 9))
		plt.suptitle(name)
		# 1 subplot
		ax1 = plt.subplot(411)
		draw_slice_borders(sim_time, skin_stim_time)
		plt.plot(shared_x, voltage, color='b')
		plt.plot(sp, [min(voltage)] * len(sp), '.', color='r', alpha=0.7)
		plt.axvspan(0, 25, color='k', alpha=0.5)
		plt.xlim(0, sim_time)
		plt.ylabel("Voltage, mV")
		plt.grid()

		# 2 subplot
		plt.subplot(412, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)
		plt.plot(shared_x, g_e, color='r')
		plt.axvspan(0, 25, color='k', alpha=0.5)
		plt.ylabel("Conductance, uS / cm2")
		plt.xlim(0, sim_time)
		plt.grid()

		# 3 subplot
		plt.subplot(413, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)
		plt.plot(shared_x, g_i, color='b')
		plt.axvspan(0, 25, color='k', alpha=0.5)
		plt.ylabel("Conductance, uS / cm2")
		plt.xlim(0, sim_time)
		plt.grid()

		# 4 subplot
		plt.subplot(414, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)
		plt.hist(sp, bins=range(sim_time))   # bin is equal to 1ms
		plt.axvspan(0, 25, color='k', alpha=0.5)
		plt.xlim(0, sim_time)
		plt.ylim(bottom=0)
		plt.grid()
		plt.ylabel("Spikes, n")
		plt.ylim(bottom=0)

		plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.08)
		plt.savefig(f"{save_to}/{name}.png", format="png", dpi=200)
		# plt.show()
		plt.close()
		logger.info(name)


def run():
	step = 0.025
	skin_stim_time = 25

	if len(sys.argv) == 2:
		path = sys.argv[1]
	else:
		path = "/home/alex/GitHub/memristive-spinal-cord/GRAS/gras_neuron/dat/"

	plot(skin_stim_time, *read_data(path), step=step, save_to=f"{path}/results/")

if __name__ == "__main__":
	run()
