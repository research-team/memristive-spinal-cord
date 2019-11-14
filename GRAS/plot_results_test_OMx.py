import os
import sys
import pylab as plt
import os
import logging
import subprocess
import numpy as np
import h5py as hdf5
# from home.kseniia.memristive-spinal-cord-STDP.GRAS import plot_shadows_boxplot
from GRAS.shadows_boxplot import plot_shadows_boxplot
import logging as log
from matplotlib.ticker import MaxNLocator

tests_number, cms, ees, inh, ped, ht5, save_all = range(7)



log.basicConfig(format='[%(funcName)s]: %(message)s', level=log.INFO)
logger = log.getLogger()

def run_tests(build_folder, args):
	"""
	Run N-times cpp builded openMP file via bash commands
	Args:
		build_folder (str): where cpp file is placed
		args (dict): special args for building properly simulation
	"""
	# buildname = "build"
	assert args[ped] in [2, 4]
	assert args[ht5] in [0, 1]
	assert 0 <= args[inh] <= 100
	assert args[cms] in [21, 15, 6]

	gcc = "/usr/bin/g++"
	buildfile = "openmp.cpp"

	for itest in range(args[tests_number]):
		logger.info(f"running test #{itest}")
		cmd_build = f"{gcc} -fopenmp {build_folder}/{buildfile}"
		cmd_run = f"./a.out {args[cms]} {args[ees]} {args[inh]} {args[ped]} {args[ht5]} {args[save_all]} {itest}"

		for cmd in [cmd_build, cmd_run]:
			logger.info(f"Execute: {cmd}")
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			out, err = process.communicate()

			for output in str(out.decode("UTF-8")).split("\n"):
				logger.info(output)
			for error in str(err.decode("UTF-8")).split("\n"):
				logger.info(error)


def convert_to_hdf5(result_folder, args):
	"""
	Converts dat files into hdf5 with compression
	Args:
		result_folder (str): folder where is the dat files placed
	"""
	# process only files with these muscle names
	for muscle in ["MN_E", "MN_F"]:
		logger.info(f"converting {muscle} dat files to hdf5")
		is_datfile = lambda f: f.endswith(f"{muscle}.dat")
		datfiles = filter(is_datfile, os.listdir(result_folder))

		# prepare hdf5 file for writing data per test
		name = f"gras_{muscle.replace('MN_', '')}_" \
			f"{args[cms]}cms_" \
			f"{args[ees]}Hz_" \
			f"i{args[inh]}_" \
			f"{args[ped]}pedal_" \
			f"{'' if args[ht5] else 'no'}5ht_T_0.025step" \
			f".hdf5"

		with hdf5.File(f"{result_folder}/{name}", 'w') as hdf5_file:
			for test_index, filename in enumerate(datfiles):
				with open(f"{result_folder}/{filename}") as datfile:
					data = [-float(v) for v in datfile.readline().split()]
					# check on NaN values (!important)
					if any(map(np.isnan, data)):
						logging.info(f"{filename} has NaN... skip")
						continue
					hdf5_file.create_dataset(f"{test_index}", data=data, compression="gzip")
		# check that hdf5 file was written properly
		with hdf5.File(f"{result_folder}/{name}") as hdf5_file:
			assert all(map(len, hdf5_file.values()))


def plot_results1(save_folder, ees_hz=40, sim_step=0.025):
	"""
	Plot hdf5 results by invoking special function of plotting shadows based on boxplots
	Args:
		save_folder (str): folder of hdf5 results and folder for saving current plots
		ees_hz (int): value of EES in Hz
		sim_step (float): simulation step (0.025 is standard)
	"""
	# for each hdf5 file get its data and plot
	for filename in filter(lambda f: f.endswith(".hdf5"), os.listdir(save_folder)):
		title = os.path.splitext(filename)[0]
		logging.info(f"start plotting {filename}")
		with hdf5.File(f"{save_folder}/{filename}") as hdf5_file:
			listed_data = np.array([data[:] for data in hdf5_file.values()])
			from time import time
			start = time()
			plot_shadows_boxplot(listed_data, ees_hz, sim_step, save_folder=save_folder, filename=title)
			end = time()
			print(end - start)


def testrunner():
	script_place = "/home/kseniia/Desktop/memristive-spinal-cord-21b254b0ab8d5e24a01be62ede06d82d08c147fa/GRAS"
	save_folder = f"{script_place}/dat"

	args = {tests_number: 1,
	        cms: 21,
	        ees: 40,
	        inh: 100,
	        ped: 2,
	        ht5: 0,
	        save_all: 1}

	run_tests(script_place, args)
	convert_to_hdf5(save_folder, args)
	plot_results1(save_folder, ees_hz=args[ees])


	if args.get(save_all) == 1 and args.get(tests_number) == 1:
		run()


def read_data(path):
	names = []
	voltage = []
	g_exc = []
	g_inh = []
	spikes = []

	for filename in filter(lambda f: f.endswith(".dat"), sorted(os.listdir(path))):
		with open(f"{path}/{filename}") as file:
			names.append(filename.replace(".dat", ""))
			voltage.append(list(map(float, file.readline().split())))
			g_exc.append(list(map(float, file.readline().split())))
			g_inh.append(list(map(float, file.readline().split())))
			spikes.append(list(map(float, file.readline().split())))
	logger.info(f"done: {path}")

	return names, voltage, g_exc, g_inh, spikes

import numpy as np

def draw_slice_borders(sim_time, skin_stim_time, extensor_first=True):
	step_cycle = 11 * skin_stim_time

	if extensor_first:
		step_pack = np.array([6 * skin_stim_time, step_cycle])
	else:
		step_pack = np.array([5 * skin_stim_time, step_cycle])

	for _ in range(int(sim_time / step_cycle)):
		plt.axvline(x=step_pack[0], linewidth=3, color='k')
		plt.axvline(x=step_pack[1], linewidth=3, color='k')
		step_pack += step_cycle

	for i in range(0, sim_time, skin_stim_time):
		plt.axvline(x=i, linewidth=1, color='k')

def plot(skin_stim_time, names, voltages, g_exc, g_inh, spikes, step, save_to, plot_only=None):
	for name, voltage, g_e, g_i, s in zip(names, voltages, g_exc, g_inh, spikes):
		if plot_only and name != plot_only:
			continue
		if not("OM1" in name or "MN_E" in name or "MN_F" in name): #or "MN_E" in name or "MN_F" in name
			continue
			
		sim_time = int(len(voltage) * step)
		slices_number = int(len(voltage) * step / 25)
		shared_x = list(map(lambda x: x * step, range(len(voltage))))

		if "MN_E" in name or "MN_F" in name:
			plt.figure(figsize=(16, 9))
			plt.suptitle(f"Sliced {name}")
			V_rest = 72

			for slice_index in range(slices_number):
				chunk_start = int(slice_index * 25 / step)
				chunk_end = int((slice_index + 1) * 25 / step)

				Y = [-v - V_rest for v in voltage[chunk_start:chunk_end]]
				X = [x * step for x in range(len(Y))]
				shifted_y = [y + 50 * slice_index for y in Y]

				plt.plot(X, shifted_y, linewidth=0.7, color='r')

			for t in [13, 15, 17, 21]:
				plt.axvline(t, color='gray', linewidth=2, linestyle='--')

			plt.xlim(0, 25)
			plt.yticks([])
			plt.savefig(f"{save_to}/sliced_{name}.png", format="png", dpi=200)
			plt.close()

		plt.figure(figsize=(16, 9))
		plt.suptitle(name)

		# 1
		ax1 = plt.subplot(311)
		draw_slice_borders(sim_time, skin_stim_time)

		plt.plot(shared_x, voltage, color='b')
		plt.plot(s, [0] * len(s), '.', color='r', alpha=0.7)

		plt.xlim(0, sim_time)
		plt.ylim(-100, 60)
		# a small hotfix to hide x lables but save x ticks -- set them as white color
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.ylabel("Voltage, mV")

		plt.grid()

		# 2
		ax2 = plt.subplot(312, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)

		plt.plot(shared_x, g_e, color='r')
		plt.plot(shared_x, g_i, color='b')
		ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
		plt.ylabel("Currents, pA")

		plt.ylim(bottom=0)
		plt.xlim(0, sim_time)
		plt.xticks(range(0, sim_time + 1, 5 if sim_time <= 275 else 25), [""] * sim_time, color='w')
		plt.grid()

		# 3
		plt.subplot(313, sharex=ax1)
		draw_slice_borders(sim_time, skin_stim_time)

		plt.hist(s, bins=range(sim_time))   # bin is equal to 1ms
		plt.xlim(0, sim_time)
		plt.ylim(bottom=0)
		plt.grid()

		plt.ylabel("Spikes, n")
		plt.ylim(bottom=0)
		ticks = range(0, sim_time + 1, 5 if sim_time <= 275 else 25)
		plt.xticks(ticks, ticks, fontsize=8)

		plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.08)

		plt.savefig(f"{save_to}/{name}.png", format="png", dpi=200)
		plt.close()

		logger.info(name)


def run():
	step = 0.025
	skin_stim_time = 25
	nuclei = None

	if len(sys.argv) == 2:
		path = sys.argv[1]
	elif len(sys.argv) == 3:
		path = sys.argv[1]
		nuclei = sys.argv[2]
	else:
		path = "/home/kseniia/Desktop/memristive-spinal-cord-21b254b0ab8d5e24a01be62ede06d82d08c147fa/GRAS/dat"

	plot(skin_stim_time, *read_data(path), step=step, save_to=f"{path}/OM1/", plot_only=nuclei) #must change path! /OMx/


if __name__ == "__main__":
	testrunner()
	run()
	
