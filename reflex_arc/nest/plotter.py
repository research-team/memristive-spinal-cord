import pylab
import os
from pkg_resources import resource_filename

names = ['moto', 'ia', 'ii', 'in']


def plot_one(name):
    results = dict()
    for file in os.listdir('results'):
        if name in file:
            with open(os.path.join('results', file)) as data:
                for line in data:
                    time, value = line.split()[1:]
                    results[float(time)] = float(value)

    x = sorted(results.keys())
    y = [results[key] for key in x]
    pylab.plot(x, y, 'r')
    pylab.ylabel(name)


def plot():
    for i in range(4):
        pylab.subplot(4, 1, i + 1)
        plot_one(names[i-1])
    # pylab.show()
    pylab.subplots_adjust(hspace=0.4)
    pylab.savefig('result.png', dpi=120)
