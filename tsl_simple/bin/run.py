import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from tsl_simple.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from tsl_simple.src.topology import Topology
from tsl_simple.src.params import simulation_time
topology = Topology()
nest.Simulate(simulation_time)

from tsl_simple.src.tools.plotter import Plotter

Plotter.plot_all_spikes()
Plotter.plot_all_voltages()
