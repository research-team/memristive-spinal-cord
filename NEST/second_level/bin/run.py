import os
import sys
import nest
import getopt
import logging

# Add the work directory to the PATH
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))
logging.basicConfig(level=logging.DEBUG)

from NEST.second_level.src.tools.cleaner import Cleaner
from NEST.second_level.src.paths import topologies_path
from NEST.second_level.src.tools.plotter import Plotter
from NEST.second_level.src.tools.miner import Miner
from NEST.second_level.src.namespace import *
from NEST.second_level.src.data import *


def simulate(simulation_params, iteration=None):
	"""
	The main function for starting simulation of the chosen topology
	Args:
		simulation_params (dict):
			parameters of the simulation
		iteration (int):
			index number of the iteration. Is need for file naming
	Returns:
		object: params of the choosen topology
	"""
	# reset NEST settings and set new one
	nest.ResetKernel()
	nest.SetKernelStatus({
		'total_num_virtual_procs': 4,
		'print_time': True,
		'resolution': 0.025,
		'overwrite_files': True})
	# import the topology via topology filename
	topology = __import__('{}.{}'.format(topologies_path, simulation_params[Params.MODEL.value]),
	                      fromlist=['Topology'])
	topology.Topology(simulation_params, test_iteration=iteration)

	# simulate the topology
	nest.Simulate(float(simulation_params[Params.SIM_TIME.value]))


def main(argv):
	"""
	The main function where it cleans the working directory,
	starts simulation and plots results
	Args:
		argv (list[str, int or optional]):
			first arg is name of the topology, the second
			is optional -- number of tests
	"""
	topology_name = argv[0]
	tests_number = int(argv[1]) if len(argv) >= 2 else 1
	is_multitest = tests_number > 1
	threshold = True
	speed = 21

	if speed == 21:
		c_time = 25
	elif speed == 15:
		c_time = 50
	elif speed == 6:
		c_time = 125
	else:
		c_time = -1

	simulation_params = {
		Params.MODEL.value: topology_name,
		Params.EES_RATE.value: 40,
		Params.RECORD_FROM.value: 'V_m',
		Params.INH_COEF.value: 1,
		Params.SPEED.value: speed,
		Params.C_TIME.value: c_time,
		Params.SIM_TIME.value: c_time * 5, # flexor 5, extensor 6
		Params.ESS_THRESHOLD.value: threshold,
		Params.MULTITEST.value: is_multitest
	}

	# clean the working folder and recreate the structure
	Cleaner.clean()
	Cleaner.create_structure()

	# simulate N times
	for test_index in range(tests_number):
		logging.info("Test number {}/{} is simulating".format(test_index + 1, tests_number))
		simulate(simulation_params, iteration=test_index)

	plotter = Plotter(simulation_params)
	# plot all nodes
	if not is_multitest:
		# plot voltages for each node
		for name in multimeters_dict.keys():
			plotter.plot_voltage(name, with_spikes=True)

	# plot slices
	plotter.plot_slices(tests_number=tests_number, from_memory=False)


if __name__ == "__main__":
	if len(sys.argv) >= 2:
		main(sys.argv[1:])
	else:
		main(["flexor_v1", 10])
