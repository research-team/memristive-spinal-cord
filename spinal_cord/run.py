import sys
sys.path.append('/home/cmen/rt-msc/')
import nest
import sys
from spinal_cord.fibers import AfferentFibers
from spinal_cord.level1 import Level1
from spinal_cord.level2 import Level2
from spinal_cord.params import Params
from spinal_cord.toolkit.plotter import clear_results
from spinal_cord.weights import init

time = 1100.
if len(sys.argv) > 1:
    time = float(sys.argv[1])
    clear_results(sys.argv[2])
    params = sys.argv[2].split('_')
    index = int(params[0])
    init(weights=params[1:])
else:
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
nest.Simulate(time)

level1.plot_motogroups()
level2.plot_pool()
# level2.plot_pc()
# level1.plot_slices(afferents.dsaf.name)
# afferents.plot_afferents()
# level1.plot_moto_only()

