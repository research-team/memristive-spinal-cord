import pylab
import os

names = dict()
names['moto'] = ['Mn_R', 'Mn_F', 'Mn_E', 'Mn_L']


def plot_one(name):
    results = dict()
    for file in os.listdir('results'):
        if name.lower() in file:
            with open(os.path.join('results', file)) as data:
                for line in data:
                    time, value = line.split()[1:]
                    results[float(time)] = float(value)

    x = sorted(results.keys())
    y = [results[key] for key in x]
    pylab.plot(x, y, 'r')
    pylab.ylabel(name)


def plot(gen_rate, glu_weight, gaba_weight, static_weight, group: str):
    for i in range(5):
        pylab.subplot(5, 1, i + 1)
        plot_one(names[group][i])
    pylab.subplots_adjust(hspace=0.4)
    pylab.savefig('R_A_2_{}_{}Hz_glu{}_gaba{}_stat{}.png'.format(group, gen_rate, glu_weight, gaba_weight, static_weight),
                  dpi=120)
    pylab.close('all')


def simple_plot(name):
    for i in range(4):
        pylab.subplot(4, 1, i + 1)
        plot_one(names[i])
    pylab.subplots_adjust(hspace=0.4)
    pylab.savefig('result{}.png'.format(name), dpi=120)
