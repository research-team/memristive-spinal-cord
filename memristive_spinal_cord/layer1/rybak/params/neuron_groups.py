from enum import Enum
import neucogar.namespaces as NEST_NAMESPACE
from memristive_spinal_cord.layer1.neuron_group_params import NeuronGroupParams

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
    )
    L_MOTOR = NeuronGroupParams(
        display_name="L Motoneurons",
        type=NEST_NAMESPACE.Glu,
    )
    R_RENSHAW = NeuronGroupParams(
        display_name="R Renshaw",
        type=NEST_NAMESPACE.GABA,
    )
    L_RENSHAW = NeuronGroupParams(
        display_name="L Renshaw",
        type=NEST_NAMESPACE.GABA,
    )
    R_INTER_1A = NeuronGroupParams(
        display_name="R 1A Interneurons",
        type=NEST_NAMESPACE.GABA,
    )
    L_INTER_1A = NeuronGroupParams(
        display_name="L 1A Interneurons",
        type=NEST_NAMESPACE.GABA,
    )
    R_INTER_1B = NeuronGroupParams(
        display_name="R 1B Interneurons",
        type=NEST_NAMESPACE.GABA,
    )
    L_INTER_1B = NeuronGroupParams(
        display_name="L 1B Interneurons",
        type=NEST_NAMESPACE.GABA,
    )
