import neucogar.namespaces as NEST_NAMESPACE
from memristive_spinal_cord.layer1.neuron_group_params import NeuronGroupParams
from memristive_spinal_cord.layer1.moraud.neuron_group_names import Layer1NeuronGroupNames

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

layer1_neuron_group_params = dict()

layer1_neuron_group_params[Layer1NeuronGroupNames.FLEX_MOTOR] = NeuronGroupParams(
    type=NEST_NAMESPACE.Glu,
    model=general_neuron_model,
    number=neuron_number_in_group,
)

layer1_neuron_group_params[Layer1NeuronGroupNames.EXTENS_MOTOR] = NeuronGroupParams(
    type=NEST_NAMESPACE.Glu,
    model=general_neuron_model,
    number=neuron_number_in_group,
)

layer1_neuron_group_params[Layer1NeuronGroupNames.FLEX_INTER_1A] = NeuronGroupParams(
    type=NEST_NAMESPACE.GABA,
    model=general_neuron_model,
    number=neuron_number_in_group,
)

layer1_neuron_group_params[Layer1NeuronGroupNames.EXTENS_INTER_1A] = NeuronGroupParams(
    type=NEST_NAMESPACE.GABA,
    model=general_neuron_model,
    number=neuron_number_in_group,
)

layer1_neuron_group_params[Layer1NeuronGroupNames.FLEX_INTER_2] = NeuronGroupParams(
    type=NEST_NAMESPACE.Glu,
    model=general_neuron_model,
    number=neuron_number_in_group,
)

layer1_neuron_group_params[Layer1NeuronGroupNames.EXTENS_INTER_2] = NeuronGroupParams(
    type=NEST_NAMESPACE.Glu,
    model=general_neuron_model,
    number=neuron_number_in_group,
)
