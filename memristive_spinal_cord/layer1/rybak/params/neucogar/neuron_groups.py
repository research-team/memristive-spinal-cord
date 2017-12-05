from memristive_spinal_cord.layer1.rybak.params.neuron_groups import *

general_neuron_model = {
    't_ref': [2.5, 4.0],  # Refractory period
    'V_m': -70.0,  #
    'E_L': -70.0,  #
    'E_K': -77.0,  #
    'g_L': 30.0,  #
    'g_Na': 12000.0,  #
    'g_K': 3600.0,  #
    'C_m': 134.0,  # Capacity of membrane (pF)
    'tau_syn_ex': 0.2,  # Time of excitatory action (ms)
    'tau_syn_in': 2.0  # Time of inhibitory action (ms)
}

neuron_number_in_group = 20

for group in Layer1Groups:
    group.set_model(general_neuron_model)
    group.set_number(neuron_number_in_group)
