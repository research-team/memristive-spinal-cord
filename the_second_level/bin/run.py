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

from the_second_level.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from the_second_level.src.paths	import topologies_path
# print('{}.{}'.format(topologies_path, sys.argv[1]))
topology = __import__('{}.{}'.format(topologies_path, sys.argv[1]), globals(), locals(), ['Params'], 0)
Params = topology.Params
to_plot = topology.to_plot
to_plot_with_slices = topology.to_plot_with_slices
# from the_second_level.src.topology import Params
nest.Simulate(Params.SIMULATION_TIME.value)

from the_second_level.src.tools.plotter import Plotter

for key in to_plot.keys():
	Plotter.plot_voltage(key, to_plot[key])
	Plotter.save_voltage(key)

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

for key in to_plot_with_slices.keys():
	Plotter.plot_slices(num_slices=to_plot_with_slices[key], name=key)
