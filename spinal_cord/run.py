import nest
from spinal_cord.level1 import Level1


nest.SetKernelStatus({
    'total_num_virtual_procs': 8,
    'print_time': True
})

nest.Install('research_team_models')
level1 = Level1()
nest.Simulate(100.)
