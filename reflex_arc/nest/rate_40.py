import nest
import os

from reflex_arc.nest.plotter import plot, simple_plot
import shutil

nest.ResetKernel()

nest.SetKernelStatus({"print_time": True,
                     "local_num_threads": 4})

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

glu_weight = 300.
gaba_weight = -200.
static_weight = 200.
gen_rate = 40.

glu = {'model': 'static_synapse',
      'delay': 1.,
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
nest.Connect(pre=i_a, post=m_n, conn_spec=conn_spec, syn_spec=glu)

nest.Connect(pre=i_i, post=i_n, conn_spec=conn_spec, syn_spec=glu)

nest.Connect(pre=i_n, post=m_n, conn_spec=conn_spec, syn_spec=gaba)


generators = nest.Create("poisson_generator", 2, {
                        'rate': gen_rate})

nest.Connect(pre=[generators[0]], post=i_a, syn_spec=static_syn)

nest.Connect(pre=[generators[1]], post=i_i, syn_spec=static_syn)


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
nest.Simulate(100.)

print("end of simulation")

plot(gen_rate, glu_weight, gaba_weight, static_weight)

