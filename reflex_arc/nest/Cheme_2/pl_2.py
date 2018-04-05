import pylab
import os

names = dict()
names['mn_in'] = ['Mn_E', 'Mn_F', 'In_0', 'In_1']
names['s_i'] = ['S_0', 'S_1', 'I_a_0', 'I_I_0', 'I_a_1', 'I_I_1']
names['ex'] = ['Ex_E', 'Ex_F', 'Ex_0', 'Ex_2', 'Ex_1', 'Ex_2_0']
names['r'] = ['S_l', 'S_t', 'S_h', 'S_r']


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
    pylab.savefig('R_A_2_{}_{}Hz_glu{}_gaba{}_stat{}.png'.format(group, gen_rate, glu_weight, gaba_weight,
                                                                 static_weight), dpi=120)
    pylab.close('all')


def simple_plot(name):
    for i in range(4):
        pylab.subplot(4, 1, i + 1)
        plot_one(names[i])
    pylab.subplots_adjust(hspace=0.4)
    pylab.savefig('result{}.png'.format(name), dpi=120)
