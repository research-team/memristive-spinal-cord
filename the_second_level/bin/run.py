import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})
nest.Install('research_team_models')

import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-3]))

from the_second_level.src.tools.cleaner import Cleaner
Cleaner.clean()
Cleaner.create_structure()

from the_second_level.src.topology import Topology
topology = Topology(int(sys.argv[1]))
nest.Simulate(300.)

from the_second_level.src.tools.plotter import Plotter

# sublayers_num = 3

# groups = []
# names = []
# for i in range(1, sublayers_num+1):
#     for name in ['right', 'left']:
#         groups.append('{}_{}'.format(name, i))
#     for name in ['Right', 'Left']:
#         names.append('{} {}'.format(name, i))
# colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

# for group, name, color in zip(groups, names, colors):
#     Plotter.plot_voltage(group, name, color)
# Plotter.save_voltage('voltages')

# for group, color in zip(groups, colors):
#     if Plotter.has_spikes(group):
#         Plotter.plot_spikes(group, color)
# Plotter.save_spikes('spikes')


    
for tier in range(1, 7):
    for group in range(5):
        Plotter.plot_voltage(
            'tier{}e{}'.format(tier, group),
            'Tier{}E{}'.format(tier, group))
# Plotter.plot_voltage('left_2', 'Left 2')
# Plotter.plot_voltage('right_3', 'Right 3')
    Plotter.save_voltage('Tier{}'.format(tier))

# for tier in range(1, 7):
#     for group in range(5):
#         if Plotter.has_spikes('tier{}e{}'.format(tier, group)):
#             Plotter.plot_spikes(name='tier{}e{}'.format(tier, group))
# if Plotter.has_spikes('right_2'):
#     Plotter.plot_spikes(name='right_2', colors=['red'])
# if Plotter.has_spikes('left_1'):
#     Plotter.plot_spikes(name='left_1', colors=['green'])
    # Plotter.save_spikes('tier{}_spikes'.format(tier))

Plotter.plot_voltage('moto', 'Moto')
Plotter.plot_voltage('pool', 'Pool')
Plotter.save_voltage('moto-pool')