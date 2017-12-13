import neucogar.namespaces as NEST_NAMESPACE
from neucogar.SynapseModel import SynapseModel
from memristive_spinal_cord.layer1.moraud.neuron_group_names import Layer1NeuronGroupNames
import memristive_spinal_cord.layer1.moraud.params.neucogar.neuron_groups as neuron_groups_params
import memristive_spinal_cord.layer1.moraud.afferents.afferents as afferents

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

# synapses
Glutamatergic = SynapseModel(
    "Glutamatergic",
    nest_model=NEST_NAMESPACE.STDP_SYNAPSE,
    params=stdp_glu
)
GABAergic = SynapseModel(
    "GABAergic",
    nest_model=NEST_NAMESPACE.STDP_SYNAPSE,
    params=stdp_gaba
)

weight_Glu = 185
weight_GABA = -70


def connect(neuron_network):
    """
    Args:
        neuron_network (NeuronNetwork)
    """
    # source is FLEX_INTER_1A
    neuron_network.connect(
        source=Layer1NeuronGroupNames.FLEX_INTER_1A,
        target=Layer1NeuronGroupNames.EXTENS_MOTOR,
        synapse=GABAergic,
        weight=weight_GABA
    )
    neuron_network.connect(
        source=Layer1NeuronGroupNames.FLEX_INTER_1A,
        target=Layer1NeuronGroupNames.EXTENS_INTER_1A,
        synapse=GABAergic,
        weight=weight_GABA
    )

    # source is EXTENS_INTER_1A
    neuron_network.connect(
        source=Layer1NeuronGroupNames.EXTENS_INTER_1A,
        target=Layer1NeuronGroupNames.FLEX_MOTOR,
        synapse=GABAergic,
        weight=weight_GABA
    )
    neuron_network.connect(
        source=Layer1NeuronGroupNames.EXTENS_INTER_1A,
        target=Layer1NeuronGroupNames.FLEX_INTER_1A,
        synapse=GABAergic,
        weight=weight_GABA
    )

    # source is FLEX_INTER_2
    neuron_network.connect(
        source=Layer1NeuronGroupNames.FLEX_INTER_2,
        target=Layer1NeuronGroupNames.FLEX_MOTOR,
        synapse=Glutamatergic,
        weight=weight_Glu
    )

    # source is EXTENS_INTER_2
    neuron_network.connect(
        source=Layer1NeuronGroupNames.EXTENS_INTER_2,
        target=Layer1NeuronGroupNames.EXTENS_MOTOR,
        synapse=Glutamatergic,
        weight=weight_Glu
    )

    connect_afferents(neuron_network)


def connect_afferents(neuron_network):
    """
    Args:
        neuron_network (NeuronNetwork)
    """
    number = neuron_groups_params.neuron_number_in_group
    afferent_params = afferents.AfferentsFile(
        afferents.Speed.FIFTEEN,
        afferents.Interval.TWENTY,
        number,
    )

    l_onea_params = afferent_params.create_generator_params(
        afferents.Types.ONE_A,
        afferents.Muscles.FLEX,
    )
    r_onea_params = afferent_params.create_generator_params(
        afferents.Types.ONE_A,
        afferents.Muscles.EXTENS,
    )
    l_two_params = afferent_params.create_generator_params(
        afferents.Types.TWO,
        afferents.Muscles.FLEX,
    )
    r_two_params = afferent_params.create_generator_params(
        afferents.Types.TWO,
        afferents.Muscles.EXTENS,
    )
    l_onea_generators = neuron_network.create_generator(**l_onea_params)
    r_onea_generators = neuron_network.create_generator(**r_onea_params)
    l_two_generators = neuron_network.create_generator(**l_two_params)
    r_two_generators = neuron_network.create_generator(**r_two_params)

    l_motor_nuclei = neuron_network.get_neuron_group_nuclei(Layer1NeuronGroupNames.EXTENS_MOTOR)
    r_motor_nuclei = neuron_network.get_neuron_group_nuclei(Layer1NeuronGroupNames.FLEX_MOTOR)
    l_inter1a_nuclei = neuron_network.get_neuron_group_nuclei(Layer1NeuronGroupNames.EXTENS_INTER_1A)
    r_inter1a_nuclei = neuron_network.get_neuron_group_nuclei(Layer1NeuronGroupNames.FLEX_INTER_1A)
    l_inter2_nuclei = neuron_network.get_neuron_group_nuclei(Layer1NeuronGroupNames.EXTENS_INTER_2)
    r_inter2_nuclei = neuron_network.get_neuron_group_nuclei(Layer1NeuronGroupNames.FLEX_INTER_2)

    r_onea_motor_spec = l_onea_motor_spec = dict(
        synapse=dict(
            model='static_synapse',
            weight=0.052608,
            delay={'distribution': 'normal', 'mu': 2., 'sigma': 0.03},
        ),
        connection=dict(rule='all_to_all')
    )
    l_motor_nuclei.ConnectGenerator(
        l_onea_generators,
        l_onea_motor_spec["synapse"],
        l_onea_motor_spec["connection"]
    )
    r_motor_nuclei.ConnectGenerator(
        r_onea_generators,
        r_onea_motor_spec["synapse"],
        r_onea_motor_spec["connection"]
    )

    r_onea_inter1a_spec = l_onea_inter1a_spec = dict(
        synapse=dict(
            model='static_synapse',
            weight=0.0175,
            delay={'distribution': 'normal', 'mu': 2., 'sigma': 0.03},
        ),
        connection=dict(rule='all_to_all')
    )
    l_inter1a_nuclei.ConnectGenerator(
        l_onea_generators,
        l_onea_inter1a_spec["synapse"],
        l_onea_inter1a_spec["connection"]
    )
    r_inter1a_nuclei.ConnectGenerator(
        r_onea_generators,
        r_onea_inter1a_spec["synapse"],
        r_onea_inter1a_spec["connection"]
    )

    r_two_inter1a_spec = l_two_inter1a_spec = dict(
        synapse=dict(
            model='static_synapse',
            weight=0.0175,
            delay={'distribution': 'normal', 'mu': 3., 'sigma': 0.03},
        ),
        connection=dict(rule='all_to_all')
    )
    l_inter1a_nuclei.ConnectGenerator(
        l_two_generators,
        l_two_inter1a_spec["synapse"],
        l_two_inter1a_spec["connection"]
    )
    r_inter1a_nuclei.ConnectGenerator(
        r_two_generators,
        r_two_inter1a_spec["synapse"],
        r_two_inter1a_spec["connection"]
    )

    r_two_inter2_spec = l_two_inter2_spec = dict(
        synapse=dict(
            model='static_synapse',
            weight=0.0175,
            delay={'distribution': 'normal', 'mu': 3., 'sigma': 0.03},
        ),
        connection=dict(rule='all_to_all')
    )
    l_inter2_nuclei.ConnectGenerator(
        l_two_generators,
        l_two_inter2_spec["synapse"],
        l_two_inter2_spec["connection"]
    )
    r_inter2_nuclei.ConnectGenerator(
        r_two_generators,
        r_two_inter2_spec["synapse"],
        r_two_inter2_spec["connection"]
    )

