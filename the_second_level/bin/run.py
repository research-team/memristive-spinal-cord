import os
import sys
import nest

# Add set the work directory
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from the_second_level.src.tools.cleaner import Cleaner
from the_second_level.src.paths import topologies_path
from the_second_level.src.tools.plotter import Plotter
from the_second_level.src.tools.miner import Miner

nest.ResetKernel()
nest.SetKernelStatus({
	'total_num_virtual_procs': 4,
	'print_time': True,
	'resolution': 0.1})

# Clean the previous results
Cleaner.clean()
Cleaner.create_structure()

# Topology
topology = __import__('{}.{}'.format(topologies_path, "new_cpg_concept_NEW"),
                      globals(), locals(),
                      ['Params', 'Topology'], 0)
Params = topology.Params
topology.Topology()

# Simulate the topology
nest.Simulate(Params.SIMULATION_TIME.value)

# Visualie results
to_plot = Params.TO_PLOT.value
to_plot_with_slices = Params.TO_PLOT_WITH_SLICES.value

# Parallelize


from multiprocessing import Process


for name in to_plot:
	Plotter.plot_voltage(name, name)
	Plotter.save_voltage(name)

for key in to_plot_with_slices.keys():
	Plotter.plot_slices(num_slices=to_plot_with_slices[key], name=key)

# for i in range(Params.NUM_SUBLEVELS.value):
#   Plotter.plot_voltage('left{}'.format(i), 'Left {}'.format(i))
#   Plotter.save_voltage('summary')

# Build the slices
for name in to_plot:
	Miner.gather_spiketimes(name)

"""
import nest
nest.ResetKernel()
nest.SetKernelStatus({
    'total_num_virtual_procs': 2,
    'print_time': True,
    'resolution': 0.1})

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))


from the_second_level.src.paths	import topologies_path
topology = __import__('{}.{}'.format(topologies_path, sys.argv[1]), globals(), locals(), ['Params', 'Topology'], 0)
Params = topology.Params
topology.Topology()


from the_second_level.src.tools.plotter import Plotter

to_plot = Params.TO_PLOT.value
to_plot_with_slices = Params.TO_PLOT_WITH_SLICES.value


for key in to_plot_with_slices.keys():
	Plotter.plot_slices(num_slices=to_plot_with_slices[key], name=key)
Plotter.save_voltage('slices{}Hz-{}Inh-{}sublevels'.format(Params.RATE.value, Params.INH_COEF.value, Params.NUM_SUBLEVELS.value))

# for i in range(Params.NUM_SUBLEVELS.value):
# 	Plotter.plot_voltage('left{}'.format(i), 'Left {}'.format(i))
# Plotter.save_voltage('summary')

from the_second_level.src.tools.miner import Miner

for name in to_plot:
	Miner.gather_spiketimes(name)

"""
