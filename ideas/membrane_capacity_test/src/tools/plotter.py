import pylab
import os
from pkg_resources import resource_filename


def plot(name: str, capacity: float) -> None:
    times = []
    values = []
    results_dir = resource_filename('membrane_capacity_test', 'results')
    for result_file in os.listdir(os.path.join(results_dir, 'raw_data')):
        if name in result_file:
            with open(os.path.join(results_dir, 'raw_data', result_file)) as data:
                for line in data:
                    time, value = line.split()
                    times.append(time)
                    values.append(value)
    pylab.plot(times, values, label='{} pF'.format(capacity))

def save(name: str) -> None:
    pylab.ylabel('Membrane potential, mV')
    pylab.xlabel('time, ms')
    pylab.title('Comparison of three neurons with different membrane capacity', fontsize=12)
    pylab.legend()
    results_dir = resource_filename('membrane_capacity_test', 'results')
    pylab.savefig(os.path.join(results_dir, 'img', '{}.png'.format(name)))
