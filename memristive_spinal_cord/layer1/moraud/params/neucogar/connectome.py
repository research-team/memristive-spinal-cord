import neucogar.namespaces as NEST_NAMESPACE
from memristive_spinal_cord.layer1.moraud.neuron_groups import Layer1Neurons
from memristive_spinal_cord.layer1.moraud.afferents import Layer1Afferents
from memristive_spinal_cord.layer1.params.connection_params import ConnectionParams
from memristive_spinal_cord.layer1.params.connection_params_storage import ConnectionParamsStorage

weight_Glu = 185
weight_GABA = -70

syn_stdp_glu = {
    'model': NEST_NAMESPACE.STDP_SYNAPSE,
    'delay': 2.5,  # Synaptic delay
    'alpha': 1.0,  # Coeficient for inhibitory STDP time (alpha * lambda)
    'lambda': 0.0005,  # Time interval for STDP
    'Wmax': 300,  # Maximum possible weight
    'mu_minus': 0.005,  # STDP depression step
    'mu_plus': 0.005,  # STDP potential step
    'weight': weight_Glu
}

syn_stdp_gaba = {
    'model': NEST_NAMESPACE.STDP_SYNAPSE,
    'delay': 1.25,  # Synaptic delay
    'alpha': 1.0,  # Coeficient for inhibitory STDP time (alpha * lambda)
    'lambda': 0.0075,  # Time interval for STDP
    'Wmax': -300.0,  # Maximum possible weight
    'mu_minus': 0.005,  # STDP depression step
    'mu_plus': 0.005,  # STDP potential step
    'weight': weight_GABA
}

conn_one_to_one = {
    'rule': 'one_to_one'
}
conn_all_to_all = {
    'rule': 'all_to_all'
}

params_storage = ConnectionParamsStorage()

# source is FLEX_INTER_1A
params_storage.add(
    ConnectionParams(
        pre=Layer1Neurons.FLEX_INTER_1A,
        post=Layer1Neurons.EXTENS_MOTOR,
        syn_spec=syn_stdp_gaba,
        conn_spec=conn_one_to_one,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Neurons.FLEX_INTER_1A,
        post=Layer1Neurons.EXTENS_INTER_1A,
        syn_spec=syn_stdp_gaba,
        conn_spec=conn_one_to_one,
    )
)

# source is EXTENS_INTER_1A
params_storage.add(
    ConnectionParams(
        pre=Layer1Neurons.EXTENS_INTER_1A,
        post=Layer1Neurons.FLEX_MOTOR,
        syn_spec=syn_stdp_gaba,
        conn_spec=conn_one_to_one,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Neurons.EXTENS_INTER_1A,
        post=Layer1Neurons.FLEX_INTER_1A,
        syn_spec=syn_stdp_gaba,
        conn_spec=conn_one_to_one,
    )
)

# source is FLEX_INTER_2
params_storage.add(
    ConnectionParams(
        pre=Layer1Neurons.FLEX_INTER_2,
        post=Layer1Neurons.FLEX_MOTOR,
        syn_spec=syn_stdp_glu,
        conn_spec=conn_one_to_one,
    )
)

# source is EXTENS_INTER_2
params_storage.add(
    ConnectionParams(
        pre=Layer1Neurons.EXTENS_INTER_2,
        post=Layer1Neurons.EXTENS_MOTOR,
        syn_spec=syn_stdp_glu,
        conn_spec=conn_one_to_one,
    )
)

######## Afferents ##########

generator_1a_motor_syn_spec = dict(
    model='static_synapse',
    weight=0.052608,
    delay={'distribution': 'normal', 'mu': 2., 'sigma': 0.03},
)
generator_1a_inter1a_syn_spec = dict(
    model='static_synapse',
    weight=0.0175,
    delay={'distribution': 'normal', 'mu': 2., 'sigma': 0.03},
)
generator_2_inter1a_syn_spec = dict(
    model='static_synapse',
    weight=0.0175,
    delay={'distribution': 'normal', 'mu': 3., 'sigma': 0.03},
)
generator_2_inter2_syn_spec = dict(
    model='static_synapse',
    weight=0.0175,
    delay={'distribution': 'normal', 'mu': 3., 'sigma': 0.03},
)

# source is FLEX_1A
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_1A,
        post=Layer1Neurons.FLEX_MOTOR,
        syn_spec=generator_1a_motor_syn_spec,
        conn_spec=conn_one_to_one,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_1A,
        post=Layer1Neurons.FLEX_INTER_1A,
        syn_spec=generator_1a_inter1a_syn_spec,
        conn_spec=conn_one_to_one,
    )
)

# source is EXTENS_1A
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.EXTENS_1A,
        post=Layer1Neurons.EXTENS_MOTOR,
        syn_spec=generator_1a_motor_syn_spec,
        conn_spec=conn_one_to_one,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.EXTENS_1A,
        post=Layer1Neurons.EXTENS_INTER_1A,
        syn_spec=generator_1a_inter1a_syn_spec,
        conn_spec=conn_one_to_one,
    )
)

# source is FLEX_2
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_2,
        post=Layer1Neurons.FLEX_INTER_1A,
        syn_spec=generator_2_inter1a_syn_spec,
        conn_spec=conn_one_to_one,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_2,
        post=Layer1Neurons.FLEX_INTER_2,
        syn_spec=generator_2_inter2_syn_spec,
        conn_spec=conn_one_to_one,
    )
)

# source is EXTENS_2
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.EXTENS_2,
        post=Layer1Neurons.EXTENS_INTER_1A,
        syn_spec=generator_2_inter1a_syn_spec,
        conn_spec=conn_one_to_one,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.EXTENS_2,
        post=Layer1Neurons.EXTENS_INTER_2,
        syn_spec=generator_2_inter2_syn_spec,
        conn_spec=conn_one_to_one,
    )
)
