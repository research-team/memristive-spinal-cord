import os
import sys
import nest
import getopt

# Add the work directory to the PATH
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from the_second_level.src.tools.cleaner import Cleaner
from the_second_level.src.paths import topologies_path
from the_second_level.src.tools.plotter import Plotter
from the_second_level.src.tools.miner import Miner


def cleaner():
	"""Clean the previous results"""
	Cleaner.clean()
	Cleaner.create_structure()

def simulate(topology_name, multitest=False, iteration=None):
	"""
	Args:
		topology_name:
		multitest:
		iteration:
	Returns:
		class:  Params class
	"""
	nest.ResetKernel()
	nest.SetKernelStatus({
		'total_num_virtual_procs': 4,
		'print_time': True,
		'resolution': 0.1,
		'overwrite_files': True})
	# create topology
	topology = __import__('{}.{}'.format(topologies_path, topology_name),
	                      globals(), locals(),
	                      ['Params', 'Topology'], 0)
	Params = topology.Params
	topology.Topology(multitest=multitest, iteration=iteration)

	# simulate the topology
	nest.Simulate(Params.SIMULATION_TIME.value)

	return Params


def plot_results(params):
	"""
	Plotting results
	Args:
		params:
	"""

	to_plot = params.TO_PLOT.value
	to_plot_with_slices = params.TO_PLOT_WITH_SLICES.value

	# plot voltages for each node
	for name in to_plot:
		Plotter.plot_voltage(name, name, with_spikes=True)
		Plotter.save_voltage(name)

	# plot slices
	for key in to_plot_with_slices.keys():
		Plotter.plot_slices(num_slices=to_plot_with_slices[key], name=key)

def single_simulation(topology_name):
	cleaner()
	params = simulate(topology_name, multitest=False)
	plot_results(params)

def multi_simulation(topology_name):
	"""

	Args:
		topology_name:
	"""
	#cleaner()
	#for test_number in range(10):
	#	print("Test number", test_number)
	#	simulate(topology_name, multitest=True, iteration=test_number)
	## plot slices
	Plotter.plot_10test(num_slices=6, name="moto", plot_mean=False, from_memory=False)


def main(argv):
	topology_name = argv[0]
	if len(argv) >= 2:
		is_test = argv[1] == "test"
	else:
		is_test = False

	if is_test:
		multi_simulation(topology_name)
	else:
		single_simulation(topology_name)


if __name__ == "__main__":
	main(sys.argv[1:])
