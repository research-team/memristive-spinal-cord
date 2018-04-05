import nest
nest.SetKernelStatus({
    'total_num_virtual_procs': 7,
    'print_time': True,
    'resolution': 0.1})

import sys
sys.path.append('/home/cmen/Code/road-to-heaven/un/lab/nest_practice/')

from membrane_capacity_test.src.tools.cleaner import clean
clean()
import membrane_capacity_test.src.topology
from membrane_capacity_test.src.tools.multimeter import add_multimeter
nest.Simulate(100.)
from membrane_capacity_test.src.tools.plotter import plot, save
plot(name='low', capacity=100)
plot(name='middle', capacity=200)
plot(name='high', capacity=300)
save('summary')
