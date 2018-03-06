import nest
from spinal_cord_with_one_neuron.fibers import AfferentFibers
from spinal_cord_with_one_neuron.level1 import Level1
from spinal_cord_with_one_neuron.level2 import Level2
from spinal_cord_with_one_neuron.toolkit.plotter import clear_results


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

nest.Simulate(150.)

level1.plot_motogroups()
level2.plot_pool()
level2.plot_pc()
level2.plot_slices(afferents.dsaf.name)

