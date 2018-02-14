from memristive_spinal_cord.proposed_scheme.moraud.entities import Layer1NeuronGroups

general_neuron_model = {
    # 't_ref': [2.5, 4.0],  # Refractory period
    't_ref': 2.5,  # Refractory period
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

neuron_group_params = dict()

nest_neuron_model = "hh_cond_exp_traub"

neuron_group_params[Layer1NeuronGroups.FLEX_MOTOR] = dict(
    model=nest_neuron_model,
    params=general_neuron_model,
    n=neuron_number_in_group,
)

neuron_group_params[Layer1NeuronGroups.EXTENS_MOTOR] = dict(
    model=nest_neuron_model,
    params=general_neuron_model,
    n=neuron_number_in_group,
)

neuron_group_params[Layer1NeuronGroups.FLEX_INTER_1A] = dict(
    model=nest_neuron_model,
    params=general_neuron_model,
    n=neuron_number_in_group,
)

neuron_group_params[Layer1NeuronGroups.EXTENS_INTER_1A] = dict(
    model=nest_neuron_model,
    params=general_neuron_model,
    n=neuron_number_in_group,
)

neuron_group_params[Layer1NeuronGroups.FLEX_INTER_2] = dict(
    model=nest_neuron_model,
    params=general_neuron_model,
    n=neuron_number_in_group,
)

neuron_group_params[Layer1NeuronGroups.EXTENS_INTER_2] = dict(
    model=nest_neuron_model,
    params=general_neuron_model,
    n=neuron_number_in_group,
)
