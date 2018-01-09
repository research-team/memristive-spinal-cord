from memristive_spinal_cord.layer1.moraud.neuron_groups import Layer1Neurons
from memristive_spinal_cord.layer1.moraud.afferents import Layer1Afferents
from memristive_spinal_cord.layer1.params.connection_params import ConnectionParams
from memristive_spinal_cord.layer1.params.connection_params_storage import ConnectionParamsStorage

distr_normal_2 = {'distribution': 'normal', 'mu': 2.0, 'sigma': 0.175},  # 0.175^2 = 0.03
distr_normal_3 = {'distribution': 'normal', 'mu': 3.0, 'sigma': 0.175},  # 0.175^2 = 0.03
syn_default_model = 'static_synapse'

conn_all_to_all = {
    'rule': 'all_to_all'
}

syn_spec_afferent1a_moto = {
    'model': syn_default_model,
    'delay': distr_normal_2,
    'weight': 0.052608
}
syn_spec_afferent1a_inter1a = {
    'model': syn_default_model,
    'delay': distr_normal_2,
    'weight': 0.0175
}
syn_spec_afferent2_inter2 = {
    'model': syn_default_model,
    'delay': distr_normal_3,
    'weight': 0.0175
}
syn_spec_afferent2_inter1a = syn_spec_afferent2_inter2

# TODO check that indegree has a uniform distribution
conn_spec_afferent1a_moto = conn_all_to_all
conn_spec_afferent1a_inter1a = {
    'rule': 'fixed_indegree',
    'indegree': 62,
}
conn_spec_afferent2_inter2 = {
    'rule': 'fixed_indegree',
    'indegree': 62,
}
conn_spec_afferent2_inter1a = conn_spec_afferent2_inter2

params_storage = ConnectionParamsStorage()

# source is Layer1Afferents.FLEX_1A,
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_1A,
        post=Layer1Neurons.FLEX_MOTOR,
        syn_spec=syn_spec_afferent1a_moto,
        conn_spec=conn_spec_afferent1a_moto,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_1A,
        post=Layer1Neurons.FLEX_INTER_1A,
        syn_spec=syn_spec_afferent1a_inter1a,
        conn_spec=conn_spec_afferent1a_inter1a,
    )
)

# source is Layer1Afferents.FLEX_2
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_2,
        post=Layer1Neurons.FLEX_INTER_2,
        syn_spec=syn_spec_afferent2_inter2,
        conn_spec=conn_spec_afferent2_inter2,
    )
)
params_storage.add(
    ConnectionParams(
        pre=Layer1Afferents.FLEX_2,
        post=Layer1Neurons.FLEX_INTER_1A,
        syn_spec=syn_spec_afferent2_inter1a,
        conn_spec=conn_spec_afferent2_inter1a,
    )
)
