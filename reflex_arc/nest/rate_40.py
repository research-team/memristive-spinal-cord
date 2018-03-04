import nest
import os

from reflex_arc.nest.plotter import plot, simple_plot
import shutil

nest.ResetKernel()

nest.SetKernelStatus({"print_time": True,
                     "local_num_threads": 4,
                      "resolution": 0.001})

T = 100.

nrn_parameters = {
    't_ref': 7.,  # Refractory period
    'V_m': -70.0,  #
    'E_L': -70.0,  #
    'E_K': -77.0,  #
    'g_L': 30.0,  #
    'g_Na': 12000.0,  #
    'g_K': 3600.0,  #
    'C_m': 134.0,  # Capacity of membrane (pF)
    'tau_syn_ex': 0.2,  # Time of excitatory action (ms)
    'tau_syn_in': 2.0}  # Time of inhibitory action (ms)

conn_spec = {'rule': 'one_to_one'}

glu_weight = 200.
gaba_weight = -300.
static_weight = 200.
gen_rate = 40.

glu = {'model': 'static_synapse',
      'delay': 1.,
      'weight': glu_weight}

glu_1 = {'model': 'static_synapse',
      'delay': 2.,
      'weight': glu_weight}

gaba = {'model': 'static_synapse',
        'delay': 1.,
        'weight': gaba_weight}

static_syn = {'weight': static_weight,
              'delay': 1.}


#Neurons
i_a = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

i_i = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

i_n = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

m_n = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)


#Conectomes
nest.Connect(pre=i_a, post=m_n, conn_spec=conn_spec, syn_spec=glu_1)

nest.Connect(pre=i_i, post=i_n, conn_spec=conn_spec, syn_spec=glu)

nest.Connect(pre=i_n, post=m_n, conn_spec=conn_spec, syn_spec=gaba)


rate = 1 / 40
spike_times = [rate + i * rate for i in range(int(T / rate))]
generators = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]
})
print(spike_times)

nest.Connect(pre=generators, post=i_a, syn_spec=static_syn)

nest.Connect(pre=generators, post=i_i, syn_spec=static_syn)


mm_moto = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label': 'results/moto'})
mm_ia = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label': 'results/ia'})
mm_ii = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label': 'results/ii'})
mm_in = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label': 'results/in'})


nest.Connect(pre=mm_moto, post=m_n)
nest.Connect(pre=mm_ia, post=i_a)
nest.Connect(pre=mm_ii, post=i_i)
nest.Connect(pre=mm_in, post=i_n)


if os.path.isdir('results'):
    shutil.rmtree('results')
    os.mkdir('results')
else:
    os.mkdir('results')
nest.Simulate(T)


plot(gen_rate, glu_weight, gaba_weight, static_weight)

