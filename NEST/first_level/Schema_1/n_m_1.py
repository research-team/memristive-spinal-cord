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


# Create neurons
#Afferents
II_0 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
Ia_0 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
II_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
Ia_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
#Excitatory
Ex_0 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
#Inhibitory
Iai_0 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Iai_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
#Moto
Mn_E = nest.Create("hh_cond_exp_traub", 169, nrn_parameters)
Mn_F = nest.Create("hh_cond_exp_traub", 169, nrn_parameters)


#Create multimeter
#Afferents
mm_II_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ii_0'})
mm_Ia_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ia_0'})
mm_II_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ii_1'})
mm_Ia_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ia_1'})
#Excitatory
mm_Ex_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_0'})
mm_Ex_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_1'})
#Inhibitory
mm_Iai_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/iai_0'})
mm_Iai_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/iai_1'})
#Moto
mm_Mn_E = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/mn_e'})
mm_Mn_F = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/mn_f'})
