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
topology = Topology(int(sys.argv[1]))
nest.Simulate(300.)

from tsl_simple.src.tools.plotter import Plotter

Plotter.plot_voltage('right_1', 'Right 1')
Plotter.plot_voltage('left_1', 'Left 1')
Plotter.save_voltage('sublevel_1')

Plotter.plot_voltage('heft_1', 'Hidden left 1')
Plotter.plot_voltage('hight_1', 'Hidden right 1')
Plotter.save_voltage('hidden sublevel 1')

Plotter.plot_voltage('right_2', 'Right 2')
Plotter.plot_voltage('left_2', 'Left 2')
Plotter.save_voltage('sublevel_2')

if Plotter.has_spikes('right_1'):
	Plotter.plot_spikes('right_1', 'b')
if Plotter.has_spikes('left_1'):
	Plotter.plot_spikes('left_1', 'g')
if Plotter.has_spikes('hight_1'):
	Plotter.plot_spikes('hight_1', 'r')
if Plotter.has_spikes('heft_1'):
	Plotter.plot_spikes('heft_1', 'c')
Plotter.save_spikes('sublevel_1_spikes')