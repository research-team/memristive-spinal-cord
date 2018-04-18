import nest
import os


from reflex_arc.nest.Cheme_1.pl_1 import plot
import shutil

nest.ResetKernel()

nest.SetKernelStatus({"print_time": True,
                     "local_num_threads": 4,
                      "resolution": 0.1})

from reflex_arc.nest.Cheme_1.n_m_1 import *


T = 100.

glu_weight = 5.
gaba_weight = -5.
static_weight = 60.
gen_rate = 40.

glu = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight}
gaba = {'model': 'static_synapse',
        'delay': 1.,
        'weight': gaba_weight}
static_syn = {'weight': static_weight,
              'delay': 1.}


#Conect neurons
#Afferents
nest.Connect(pre=II_0, post=Ex_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=II_0, post=Iai_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=Ia_0, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=Ia_0, post=Iai_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=II_1, post=Ex_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=II_1, post=Iai_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=Ia_1, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=Ia_1, post=Iai_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
#Excitatory
nest.Connect(pre=Ex_0, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_1, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
#Inhibitory
nest.Connect(pre=Iai_0, post=Iai_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=Iai_0, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=Iai_1, post=Iai_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=Iai_1, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)


#Create generator
time_between_spikes = 1000 / gen_rate  # time between spikes
spike_times = [round(4.5 + i * time_between_spikes, 1) for i in range(int(T / time_between_spikes))]
generators = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
generator_1 = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
print(spike_times)


#Connect generator
nest.Connect(pre=generators, post=II_0, syn_spec=static_syn)
nest.Connect(pre=generators, post=Ia_0, syn_spec=static_syn)
nest.Connect(pre=generators, post=II_1, syn_spec=static_syn)
nest.Connect(pre=generators, post=Ia_1, syn_spec=static_syn)


#Connect multimeter
#Afferents
nest.Connect(pre=mm_II_0, post=II_0)
nest.Connect(pre=mm_Ia_0, post=Ia_0)
nest.Connect(pre=mm_II_1, post=II_1)
nest.Connect(pre=mm_Ia_1, post=Ia_1)
#Excitatory
nest.Connect(pre=mm_Ex_0, post=Ex_0)
nest.Connect(pre=mm_Ex_1, post=Ex_1)
#Inhibitory
nest.Connect(pre=mm_Iai_0, post=Iai_0)
nest.Connect(pre=mm_Iai_1, post=Iai_1)
#Moto
nest.Connect(pre=mm_Mn_E, post=Mn_E)
nest.Connect(pre=mm_Mn_F, post=Mn_F)


if os.path.isdir('results'):
    shutil.rmtree('results')
    os.mkdir('results')
else:
    os.mkdir('results')
nest.Simulate(T)


plot(gen_rate, glu_weight, gaba_weight, static_weight, group='1')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='2')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='afferents')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='interneuron')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='moto')

