# stdp, 2 синапса, 20 сек. Результат в spinal cord. Модель ходжкина-хаксли

import nest

nest.ResetKernel()

nest.SetKernelStatus({
    "print_time": True,
    "local_num_threads": 4})

nrn_parameters = {
    't_ref': 4.0,
    'V_m': -70.0,
    'E_L': -70.0,
    'E_K': -77.0,
    'g_L': 30.0,
    'g_Na': 12000.0,
    'g_K': 3600.0,
    'C_m': 134.0,
    'tau_syn_ex': 0.2,
    'tau_syn_in': 2.0}

conn_spec = {'rule': 'all_to_all'}

ac = {'model': 'static_synapse',
      'delay': 1.,
      'weight': 300.}

gaba = {'model': 'static_synapse',
        'delay': 1.,
        'weight': -300.}

static_syn = {'weight': 100.,
              'delay': 1.}

i_a = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

moto = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)
