import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from neuron_group_filtration_test.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

import neuron_group_filtration_test.src.topology
nest.Simulate(200.)

from neuron_group_filtration_test.src.tools.plotter import Plotter
Plotter.plot_voltage('right_1', 'Right 1')
Plotter.plot_voltage('left_1', 'Left 1')
Plotter.plot_voltage('right_2', 'Right 2')
# Plotter.plot_voltage('left_2', 'Left 2')
# Plotter.plot_voltage('right_3', 'Right 3')
Plotter.save_voltage('voltages')

Plotter.plot_spikes(name='r1_spikes', colors=[[0, 0, 1]])
Plotter.plot_spikes(name='l1_spikes', colors=[[0, 1, 0]])
Plotter.plot_spikes(name='r2_spikes', colors=[[1, 0, 0]])
Plotter.save_spikes()
