import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from lcr_complex.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from lcr_complex.src.topology import Topology
from lcr_complex.src.params import simulation_time
topology = Topology()
nest.Simulate(simulation_time)

from lcr_complex.src.tools.plotter import Plotter

Plotter.plot_all_spikes()
Plotter.plot_all_voltages()
Plotter.plot_slices()
