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
# Mn_F
Ia_MnF = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

II_MnF = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

Mn_F = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

Ex_MnF = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

Iai_MnF = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

# Mn_E
Ia_MnE = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

II_MnE = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

Mn_E = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

Ex_MnE = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)

Iai_MnE = nest.Create("hh_cond_exp_traub", 1, nrn_parameters)


# Multimeter
# Mn_F
mm_Ia_MnF = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ia_mnf'})
mm_II_MnF = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ii_mnf'})
mm_Mn_F = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                      'results/mn_f'})
mm_Ex_MnF = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_mnf'})
mm_Iai_MnF = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                         'results/iai_mnf'})

# Mn_E
mm_Ia_MnE = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ia_mne'})
mm_II_MnE = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ii_mne'})
mm_Mn_E = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                      'results/mn_e'})
mm_Ex_MnE = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_mne'})
mm_Iai_MnE = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                         'results/iai_mne'})
