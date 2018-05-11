import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
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
from rybak_affs.src.params import simulation_time
nest.Simulate(simulation_time)

from rybak_affs.src.tools.plotter import Plotter

Plotter.plot_voltage('afferent', 'Ia Aff')
Plotter.save_voltage('ia_aff')

Plotter.plot_voltage('moto', 'Moto')
Plotter.save_voltage('moto')

Plotter.plot_voltage('rc', 'Moto')
Plotter.save_voltage('rc')

Plotter.plot_voltage('ia_int', 'Ia Interneurons')
Plotter.save_voltage('ia_int')