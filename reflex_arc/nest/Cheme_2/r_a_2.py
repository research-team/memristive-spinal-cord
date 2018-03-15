import nest
import os


from reflex_arc.nest.Cheme_2.pl_2 import plot
import shutil

nest.ResetKernel()

nest.SetKernelStatus({"print_time": True,
                     "local_num_threads": 4,
                      "resolution": 0.1})

from reflex_arc.nest.Cheme_2.n_m_2 import *

T = 100.

glu_weight = 5.
gaba_weight = -15.
static_weight = 60.
gen_rate = 15.

glu = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight}

gaba = {'model': 'static_synapse',
        'delay': 1.,
        'weight': gaba_weight}

static_syn = {'weight': static_weight,
              'delay': 1.}


#Conectomes
nest.Connect(pre=I_I, post=Ex_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Ex_2, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=I_I, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=In_1, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=In_1, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_1, post=In_1_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_1_1, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_1_1, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_1_1, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=I_a, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=I_a, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=S_h, post=In_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=In_2, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_2, post=In_2_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_2_1, post=In_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_2_1, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=S_r, post=Ex_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Ex_1, post=Mn_R, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=S_r, post=In_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=In_3, post=Mn_L, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_3, post=In_3_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_3_1, post=In_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=In_3_1, post=Mn_R, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=S, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Ex_3, post=Ex_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=Ex_4, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=Ex_4, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=Ex_4, post=Ex_4_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=Ex_4_1, post=Ex_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=Ex_4_1, post=In_1_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=Ex_4_1, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=Ex_3, post=Ex_4_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=S, post=Ex, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=S_1, post=Ex_3_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Ex_3_1, post=Ex_4_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=Ex_3_1, post=Ex_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=S_l, post=Ex_1_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Ex_1_1, post=Mn_L, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=S_l, post=In_3_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=S_t, post=In_2_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=I_a_1, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=I_a_1, post=In_1_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=I_I_1, post=Ex_2_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Ex_2_1, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)

nest.Connect(pre=I_I_1, post=In_1_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

time_between_spikes = 1000 / gen_rate  # time between spikes
spike_times = [round(4.5 + i * time_between_spikes, 1) for i in range(int(T / time_between_spikes))]
generators = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
generator_1 = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
print(spike_times)


nest.Connect(pre=generators, post=S_1, syn_spec=static_syn)

nest.Connect(pre=generators, post=S_l, syn_spec=static_syn)

nest.Connect(pre=generators, post=S_t, syn_spec=static_syn)

nest.Connect(pre=generators, post=I_a_1, syn_spec=static_syn)

nest.Connect(pre=generators, post=I_I_1, syn_spec=static_syn)

nest.Connect(pre=generators, post=I_I, syn_spec=static_syn)

nest.Connect(pre=generators, post=I_a, syn_spec=static_syn)

nest.Connect(pre=generators, post=S_h, syn_spec=static_syn)

nest.Connect(pre=generators, post=S_r, syn_spec=static_syn)

nest.Connect(pre=generators, post=S, syn_spec=static_syn)

# Mn_F
nest.Connect(pre=mm_Mn_R, post=Mn_R)
nest.Connect(pre=mm_Mn_F, post=Mn_F)
nest.Connect(pre=mm_Mn_E, post=Mn_E)
nest.Connect(pre=mm_Mn_L, post=Mn_L)


if os.path.isdir('results'):
    shutil.rmtree('results')
    os.mkdir('results')
else:
    os.mkdir('results')
nest.Simulate(T)

plot(gen_rate, glu_weight, gaba_weight, static_weight, group='moto')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='sensory')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='afferent')
