import pylab
import os

names = dict()
names['1'] = ['Iai_0', 'Mn_E', 'Ia_0', 'Ex_0', 'II_0']
names['2'] = ['Iai_1', 'Mn_F', 'Ia_1', 'Ex_1', 'II_1']


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
    for i in range(len(names[group])):
        pylab.subplot(len(names[group]), 1, i + 1)
        plot_one(names[group][i])
    pylab.subplots_adjust(hspace=0.5 + 0.02 * len(names[group]))
    pylab.savefig('R_A_1_{}_{}Hz_glu{}_gaba{}_stat{}.png'.format(group, gen_rate, glu_weight, gaba_weight,
                                                                 static_weight), dpi=120)
    pylab.close('all')
