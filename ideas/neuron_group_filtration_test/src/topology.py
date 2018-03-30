import nest
import os
from neuron_group_filtration_test.src.tools.multimeter import add_multimeter
from neuron_group_filtration_test.src.tools.spike_detector import add_spike_detector
from neuron_group_filtration_test.src.paths import raw_data_path

right_1 = nest.Create(
    model='hh_cond_exp_traub',
    n=20,
    params={'C_m': 200., 'V_m': -70., 'E_L': -70., 'tau_syn_ex': .5, 'tau_syn_in': 1.1, 't_ref': 2.})
left_1 = nest.Create(
    model='hh_cond_exp_traub',
    n=20,
    params={'C_m': 100., 'V_m': -70., 'E_L': -70., 'tau_syn_ex': .5, 'tau_syn_in': 1.1, 't_ref': 2.})
right_2 = nest.Create(
    model='hh_cond_exp_traub',
    n=20,
    params={'C_m': 200., 'V_m': -70., 'E_L': -70., 'tau_syn_ex': .5, 'tau_syn_in': 1.1, 't_ref': 2.})
left_2 = nest.Create(
    model='hh_cond_exp_traub',
    n=20,
    params={'C_m': 100., 'V_m': -70., 'E_L': -70., 'tau_syn_ex': .5, 'tau_syn_in': 1.1, 't_ref': 2.})
right_3 = nest.Create(
    model='hh_cond_exp_traub',
    n=20,
    params={'C_m': 200., 'V_m': -70., 'E_L': -70., 'tau_syn_ex': .5, 'tau_syn_in': 1.1, 't_ref': 2.})

n_spikes = 6
spike_times = [10. + i * 25. for i in range(n_spikes)]
ees = nest.Create(
    model='spike_generator',
    n=1,
    params={
        'spike_times': spike_times,
        'spike_weights': [30. for _ in spike_times]})

nest.Connect(
    pre=right_1,
    post=left_1,
    syn_spec={
        'model': 'static_synapse',
        'delay': 1.,
        'weight': 30.
    },
    conn_spec={
        'rule': 'fixed_outdegree',
        'outdegree': 4})
nest.Connect(
    pre=left_1,
    post=right_1,
    syn_spec={
        'model': 'static_synapse',
        'delay': 1.,
        'weight': 50.
    },
    conn_spec={
        'rule': 'fixed_outdegree',
        'outdegree': 4})
nest.Connect(
    pre=right_1,
    post=right_2,
    syn_spec={
        'model': 'static_synapse',
        'delay': 1.,
        'weight': 4.4
    },
    conn_spec={
        'rule': 'fixed_indegree',
        'indegree': 8,
        'multapses': False})
# nest.Connect(
#     pre=right_2,
#     post=right_3,
#     syn_spec={
#         'model': 'static_synapse',
#         'delay': 1.,
#         'weight': 28.
#     },
#     conn_spec={
#         'rule': 'fixed_indegree',
#         'indegree': 2})
nest.Connect(
    pre=left_2,
    post=right_2,
    syn_spec={
        'model': 'static_synapse',
        'delay': 1.,
        'weight': 0.
    },
    conn_spec={
        'rule': 'fixed_indegree',
        'indegree': 5})
nest.Connect(
    pre=right_2,
    post=left_2,
    syn_spec={
        'model': 'static_synapse',
        'delay': 1.,
        'weight': 0.
    },
    conn_spec={
        'rule': 'fixed_indegree',
        'indegree': 2})

nest.Connect(
    pre=add_multimeter('right_1'),
    post=right_1)
nest.Connect(
    pre=add_multimeter('left_1'),
    post=left_1)
nest.Connect(
    pre=add_multimeter('right_2'),
    post=right_2)
nest.Connect(
    pre=add_multimeter('left_2'),
    post=left_2)
nest.Connect(
    pre=add_multimeter('right_3'),
    post=right_3)

nest.Connect(
    pre=ees,
    post=right_1,
    syn_spec={
        'model': 'static_synapse',
        'delay': 0.1,
        'weight': 10.
    },
    conn_spec={
        'rule': 'fixed_outdegree',
        'outdegree': 20,
        'multapses': False})

nest.Connect(
    pre=right_1,
    post=add_spike_detector('r1_spikes'))
nest.Connect(
    pre=right_2,
    post=add_spike_detector('r2_spikes'))
nest.Connect(
    pre=left_1,
    post=add_spike_detector('l1_spikes'))
