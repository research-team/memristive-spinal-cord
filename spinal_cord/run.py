import nest
from spinal_cord.fibers import AfferentFibers
from spinal_cord.level1 import Level1
from spinal_cord.level2 import Level2
from spinal_cord.toolkit.plotter import clear_results


clear_results()
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1
})
nest.Install('research_team_models')
afferents = AfferentFibers()
level1 = Level1()
level1.connect_afferents(afferents)
level2 = Level2(level1, afferents)

nest.Simulate(175.)

level1.plot_motogroups()
level2.plot_pool()
level2.plot_pc()
level2.plot_slices(afferents.dsaf.name)
