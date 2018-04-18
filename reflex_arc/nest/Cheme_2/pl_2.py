import pylab
import os

names = dict()
names['mn_n'] = ['Mn_L', 'Mn_E', 'Mn_F', 'Mn_R', 'Noc']
names['in'] = ['In_0', 'In_1', 'In_2', 'In_3', 'In_4', 'In_5']
names['s'] = ['S_0', 'S_l', 'S_t', 'S_h', 'S_r', 'S_1']
names['af'] = ['I_a_0', 'I_I_0', 'I_I_1', 'I_a_1']
names['ex'] = ['Ex_L', 'Ex_E', 'Ex_F', 'Ex_R']
names['ex_2'] = ['Ex_1', 'Ex_2', 'Ex_3', 'Ex_4']


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
    pylab.subplots_adjust(hspace=0.5 + 0.05 * len(names[group]))
    pylab.savefig('R_A_2_{}_{}Hz_glu{}_gaba{}_stat{}.png'.format(group, gen_rate, glu_weight, gaba_weight,
                                                                 static_weight), dpi=120)
    pylab.close('all')

