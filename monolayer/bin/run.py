import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from monolayer.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from monolayer.src.topology import Topology
from monolayer.src.params import simulation_time
topology = Topology()
nest.Simulate(simulation_time)

from monolayer.src.tools.plotter import Plotter

# Plotter.plot_all_spikes()
# Plotter.plot_all_voltages()
# Plotter.plot_slices()

Plotter.plot_voltage('exc', 'Excitatory')
Plotter.plot_voltage('inh', 'Inhibitory')
Plotter.plot_voltage('aff', 'Afferents')
Plotter.save_voltage('voltages')

for name in ['exc', 'inh', 'aff']:
	if Plotter.has_spikes(name):
		Plotter.plot_spikes(name)
Plotter.save_spikes('spikes')
