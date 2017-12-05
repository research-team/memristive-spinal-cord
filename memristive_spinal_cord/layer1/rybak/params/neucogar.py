from enum import Enum
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


class Layer1Groups(Enum):
    R_MOTOR = NeuronGroupParams(
        display_name="R Motoneurons",
        type=NEST_NAMESPACE.Glu,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    L_MOTOR = NeuronGroupParams(
        display_name="L Motoneurons",
        type=NEST_NAMESPACE.Glu,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    R_RENSHAW = NeuronGroupParams(
        display_name="R Renshaw",
        type=NEST_NAMESPACE.GABA,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    L_RENSHAW = NeuronGroupParams(
        display_name="L Renshaw",
        type=NEST_NAMESPACE.GABA,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    R_INTER_1A = NeuronGroupParams(
        display_name="R 1A Interneurons",
        type=NEST_NAMESPACE.GABA,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    L_INTER_1A = NeuronGroupParams(
        display_name="L 1A Interneurons",
        type=NEST_NAMESPACE.GABA,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    R_INTER_1B = NeuronGroupParams(
        display_name="R 1B Interneurons",
        type=NEST_NAMESPACE.GABA,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
    L_INTER_1B = NeuronGroupParams(
        display_name="L 1B Interneurons",
        type=NEST_NAMESPACE.GABA,
        model=general_neuron_model,
        number=neuron_number_in_group
    )
