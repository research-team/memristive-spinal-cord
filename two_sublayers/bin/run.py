import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from two_sublayers.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from two_sublayers.src.topology import Topology
from two_sublayers.src.params import simulation_time
topology = Topology()
nest.Simulate(simulation_time)

from two_sublayers.src.tools.plotter import Plotter

# Plotter.plot_all_spikes()
# Plotter.plot_all_voltages()
# Plotter.plot_slices()

Plotter.plot_voltage('right_1', 'Right1')
Plotter.plot_voltage('right_2', 'Right2')
Plotter.plot_voltage('left_1', 'Left_1')
Plotter.plot_voltage('left_2', 'Left_2')
Plotter.save_voltage('voltages')

Plotter.plot_voltage('pool', 'Pool')
Plotter.save_voltage('pool')

Plotter.plot_voltage('aff', 'Afferents')
Plotter.save_voltage('affs')

Plotter.plot_voltage('moto', 'Moto')
Plotter.save_voltage('moto')

for name in ['right_1', 'left_1', 'right_2', 'left_2']:
	if Plotter.has_spikes(name):
		Plotter.plot_spikes(name, 'k')
Plotter.save_spikes('spikes')
