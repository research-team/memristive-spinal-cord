import nest
from spinal_cord.level1 import Level1
from spinal_cord.toolkit.plotter import clear_results


clear_results()
nest.SetKernelStatus({
    'total_num_virtual_procs': 8,
    'print_time': True,
    'resolution': 0.1
})
nest.Install('research_team_models')
level1 = Level1()
nest.Simulate(100.)
level1.plot_afferents()
