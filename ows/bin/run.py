import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from ows.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from ows.src.topology import Topology
from ows.src.params import simulation_time
topology = Topology()
nest.Simulate(simulation_time)

from ows.src.tools.plotter import Plotter

# Plotter.plot_all_spikes()
# Plotter.plot_all_voltages()
Plotter.plot_slices(num_slices=7)

from ows.src.params import num_sublevels

for i in range(num_sublevels):
	for j in range(5):
		Plotter.plot_voltage('e{}{}'.format(j, i), label='E{}{}'.format(j, i))
	Plotter.save_voltage('tier{}'.format(i))
