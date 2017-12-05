import neucogar.namespaces as NEST_NAMESPACE
from memristive_spinal_cord.layer1.neuron_group_params import NeuronGroupParams

neuron_number_in_group = 20

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
stdp_glu = {'delay': 2.5,  # Synaptic delay
            'alpha': 1.0,  # Coeficient for inhibitory STDP time (alpha * lambda)
            'lambda': 0.0005,  # Time interval for STDP
            'Wmax': 300,  # Maximum possible weight
            'mu_minus': 0.005,  # STDP depression step
            'mu_plus': 0.005  # STDP potential step
            }
stdp_gaba = {'delay': 1.25,  # Synaptic delay
             'alpha': 1.0,  # Coeficient for inhibitory STDP time (alpha * lambda)
             'lambda': 0.0075,  # Time interval for STDP
             'Wmax': -300.0,  # Maximum possible weight
             'mu_minus': 0.005,  # STDP depression step
             'mu_plus': 0.005  # STDP potential step
             }

r_motor_params = l_motor_params = NeuronGroupParams(
    NEST_NAMESPACE.Glu, general_neuron_model, neuron_number_in_group
)
r_renshaw_params = l_renshaw_params = NeuronGroupParams(
    NEST_NAMESPACE.GABA, general_neuron_model, neuron_number_in_group
)
r_inter_1a_params = l_inter_1a_params = NeuronGroupParams(
    NEST_NAMESPACE.GABA, general_neuron_model, neuron_number_in_group
)
r_inter_1b_params = l_inter_1b_params = NeuronGroupParams(
    NEST_NAMESPACE.GABA, general_neuron_model, neuron_number_in_group
)
