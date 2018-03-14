import nest

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


# Neuron
# Sensory
S = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

S_r = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

S_h = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

S_1 = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

S_l = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

S_t = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

# Inhibitory
In_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

In_2 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

In_3 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

In_1_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

In_2_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

In_3_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

# Excitatory
Ex = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_2 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_1_2 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_2_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_3 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_4 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_3_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

Ex_4_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)

# I
I_a = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)

I_I = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)

I_a_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)

I_I_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)

# Multimeter
# Mn_F
mm_S = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s'})
