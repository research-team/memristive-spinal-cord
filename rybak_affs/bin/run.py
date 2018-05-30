import nest
nest.ResetKernel()
nest.SetKernelStatus({
    'total_num_virtual_procs': 2,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from rybak_affs.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

import rybak_affs.src.topology
from rybak_affs.src.params import simulation_time, num_sublevels
nest.Simulate(simulation_time)

from rybak_affs.src.tools.plotter import Plotter

# Plotter.plot_voltage('afferent', 'Ia Aff')
# Plotter.save_voltage('ia_aff')

# Plotter.plot_voltage('moto', 'Moto')
# Plotter.save_voltage('moto')

# Plotter.plot_voltage('rc', 'RC')
# Plotter.save_voltage('rc')

# Plotter.plot_voltage('ia_int', 'Ia Interneurons')
# Plotter.save_voltage('ia_int')

# Plotter.plot_voltage('pool', 'Pool')
# Plotter.save_voltage('pool')

# for i in range(num_sublevels):
# 	Plotter.plot_voltage('general_right{}'.format(i), 'Right')
# 	Plotter.plot_voltage('general_left{}'.format(i), 'Left')
# 	Plotter.save_voltage('general{}'.format(i))
# 	Plotter.plot_voltage('hidden_left{}'.format(i), 'Hidden Left')
# 	Plotter.plot_voltage('hidden_right{}'.format(i), 'Hidden Right')
# 	Plotter.plot_voltage('inh{}'.format(i), 'Inhibitory')
# 	Plotter.save_voltage('hidden{}'.format(i))

Plotter.plot_slices(num_slices=7, name='moto')
