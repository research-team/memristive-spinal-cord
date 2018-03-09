import nest
import os


from reflex_arc.nest.plotter import plot
import shutil

nest.ResetKernel()

nest.SetKernelStatus({"print_time": True,
                     "local_num_threads": 4,
                      "resolution": 0.1})

from reflex_arc.nest.neuron_multimeter import *

T = 100.

#conn_spec = {'rule': 'one_to_one'}

#conn_spec2 = {'rule': 'fixed_indegree', 'indegree': 116, 'multapses': False}

glu_weight_ia = 3.3
glu_weight_ex = 0.
gaba_weight_iai = -0.
glu_weight_ii = 3.4

glu_ia = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight_ia}

glu_ex = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight_ex}

gaba_iai = {'model': 'static_synapse',
        'delay': 1.,
        'weight': gaba_weight_iai}

glu_ii = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight_ii}

glu_weight = 5.
gaba_weight = -15.
static_weight = 60.
gen_rate = 40.

glu = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight}

gaba = {'model': 'static_synapse',
        'delay': 2.,
        'weight': gaba_weight}

static_syn = {'weight': static_weight,
              'delay': 1.}


#Conectomes
# Mn_E
nest.Connect(pre=II_MnE, post=Ex_MnE, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu_ii)

nest.Connect(pre=Ex_MnE, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu_ex)

nest.Connect(pre=II_MnE, post=Iai_MnE, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Iai_MnE, post=Iai_MnF, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=Iai_MnE, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba_iai)

nest.Connect(pre=Ia_MnE, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu_ia)

nest.Connect(pre=Ia_MnE, post=Iai_MnE, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

# Mn_F
nest.Connect(pre=II_MnF, post=Ex_MnF, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu_ii)

nest.Connect(pre=Ex_MnF, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu_ex)

nest.Connect(pre=II_MnF, post=Iai_MnF, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)

nest.Connect(pre=Iai_MnF, post=Iai_MnE, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)

nest.Connect(pre=Iai_MnF, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba_iai)

nest.Connect(pre=Ia_MnF, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu_ia)

nest.Connect(pre=Ia_MnF, post=Iai_MnF, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)


time_between_spikes = 1000 / gen_rate  # time between spikes
spike_times = [4.5 + i * time_between_spikes for i in range(int(T / time_between_spikes))]
generators = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
generator_1 = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
print(spike_times)


nest.Connect(pre=generators, post=Ia_MnE, syn_spec=static_syn)

nest.Connect(pre=generators, post=II_MnE, syn_spec=static_syn)

nest.Connect(pre=generators, post=Ia_MnF, syn_spec=static_syn)

nest.Connect(pre=generators, post=II_MnF, syn_spec=static_syn)

# Mn_F
nest.Connect(pre=mm_Ia_MnF, post=Ia_MnF)
nest.Connect(pre=mm_II_MnF, post=II_MnF)
nest.Connect(pre=mm_Mn_F, post=Mn_F)
nest.Connect(pre=mm_Ex_MnF, post=Ex_MnF)
nest.Connect(pre=mm_Iai_MnF, post=Iai_MnF)

# Mn_E
nest.Connect(pre=mm_Ia_MnE, post=Ia_MnE)
nest.Connect(pre=mm_II_MnE, post=II_MnE)
nest.Connect(pre=mm_Mn_E, post=Mn_E)
nest.Connect(pre=mm_Ex_MnE, post=Ex_MnE)
nest.Connect(pre=mm_Iai_MnE, post=Iai_MnE)


if os.path.isdir('results'):
    shutil.rmtree('results')
    os.mkdir('results')
else:
    os.mkdir('results')
nest.Simulate(T)


plot(gen_rate, glu_weight, gaba_weight, static_weight, group='flex')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='extens')

