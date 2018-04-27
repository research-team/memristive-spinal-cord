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

Plotter.plot_voltage('extensor_moto', label='Extensor')
Plotter.plot_voltage('flexor_moto', label='Flexor')
Plotter.save_voltage('motos')

Plotter.plot_voltage('flexor_ia_afferents', label='Flexor Ia Afferents')
Plotter.plot_voltage('extensor_ia_afferents', label='Extensor Ia Afferents')
Plotter.save_voltage('ia_afferents')

Plotter.plot_voltage('flexor_sensory', label='Flexor Sensory')
Plotter.plot_voltage('extensor_sensory', label='Extensor Sensory')
Plotter.save_voltage('sensories')
#
Plotter.plot_voltage('flexor_pool', 'Flexor Pool')
Plotter.plot_voltage('extensor_pool', 'Extensor Pool')
Plotter.save_voltage('pool')

from ows.src.params import num_sublevels

# for i in range(num_sublevels):
# 	for j in range(5):
# 		Plotter.plot_voltage('e{}{}'.format(j, i), label='E{}{}'.format(j, i))
# 	Plotter.save_voltage('tier{}'.format(i))
