from memristive_spinal_cord.proposed_scheme.level2.parameters import Weights
from memristive_spinal_cord.proposed_scheme.moraud.entities import Layer1NeuronGroups


distr_normal_2 = {'distribution': 'normal', 'mu': 2.0, 'sigma': 0.175}
connection_params = {}
for tier in range(1, 7):
    current_tier = 'Tier{}'.format(tier)
    connection_params[current_tier] = {
        'e0_e1': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EE.value[tier][0]
        },
        'e1_e2': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EE.value[tier][1]
        },
        'e2_e1': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EE.value[tier][2]
        },
        'e3_e4': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EE.value[tier][3]
        },
        'e4_e3': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EE.value[tier][4]
        },
        'e0_e3': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EE.value[tier][5]
        },
        'e3_i0': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EI.value[tier][0]
        },
        'i0_e1': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': -Weights.IE.value[tier][0]
        },
        'e4_i1': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.EI.value[tier][1]
        },
        'i1_e3': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': -Weights.IE.value[tier][1]
        }
    }
    if tier < 6:
        connection_params[current_tier].update({
            'e0_e0': {
                'model': 'static_synapse',
                'delay': distr_normal_2,
                'weight': Weights.TT.value[tier][0]
            },
            'e3_e0': {
                'model': 'static_synapse',
                'delay': distr_normal_2,
                'weight': Weights.TT.value[tier][1]
            }
        })
    if 0 < tier < 6:
        print(tier)
        connection_params[current_tier].update({
            'e2_e2': {
                'model': 'static_synapse',
                'delay': distr_normal_2,
                'weight': Weights.TT.value[tier][2]
            }
        })
connection_params['Tier0'] = {
    'e2_e1': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.EE.value[0][0]
    },
    'e1_e2': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.EE.value[0][1]
    },
    'e1_i0': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.EI.value[0]
    },
    'i0_e1': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': -Weights.IE.value[0]
    }
}
for tier in range(1, 7):
    connection_params['Tier{}'.format(tier)].update({
        'e2_pool': {
            'model': 'static_synapse',
            'delay': distr_normal_2,
            'weight': Weights.PE.value[tier]
        }
    })
connection_params['Tier0'].update({
    'e1_pool0': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.PE.value[0]
    },
    'e1_pool1': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': -Weights.PI.value
    }
})
connection_params['Pool0'] = {
    'Mn-E': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.PM.value['Extensor']
    },
    'Ia-E': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.PIa.value['Extensor']
    }
}
connection_params['Pool1'] = {
    'Mn-F': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.PM.value['Flexor']
    },
    'Ia-F': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.PIa.value['Flexor']
    }
}
connection_params['Mediator'] = {
    'model': 'static_synapse',
    'delay': distr_normal_2,
    'weight': Weights.MR.value
}

""" INTERCONNECTIONS """
l2_connections_list = []

for tier in range(1, 7):
    l2_connections_list.append(
        dict(
            pre='Tier{}E0'.format(tier),
            post='Tier{}E1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e0_e1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E1'.format(tier),
            post='Tier{}E2'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e1_e2'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E2'.format(tier),
            post='Tier{}E1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e2_e1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E3'.format(tier),
            post='Tier{}E4'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e3_e4'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E4'.format(tier),
            post='Tier{}E3'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e4_e3'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E0'.format(tier),
            post='Tier{}E3'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e0_e3'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E3'.format(tier),
            post='Tier{}I0'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e3_i0'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}I0'.format(tier),
            post='Tier{}E1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['i0_e1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}E4'.format(tier),
            post='Tier{}I1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e4_i1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}I1'.format(tier),
            post='Tier{}E3'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['i1_e3'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    if tier < 6:
        l2_connections_list.append(
            dict(
                pre='Tier{}E0'.format(tier),
                post='Tier{}E0'.format(tier + 1),
                syn_spec=connection_params['Tier{}'.format(tier)]['e0_e0'],
                conn_spec={'rule': 'one_to_one'}
            )
        )
        l2_connections_list.append(
            dict(
                pre='Tier{}E3'.format(tier),
                post='Tier{}E0'.format(tier + 1),
                syn_spec=connection_params['Tier{}'.format(tier)]['e3_e0'],
                conn_spec={'rule': 'one_to_one'}
            )
        )
    if 1 < tier < 6:
        l2_connections_list.append(
            dict(
                pre='Tier{}E2'.format(tier + 1),
                post='Tier{}E2'.format(tier),
                syn_spec=connection_params['Tier{}'.format(tier)]['e2_e2'],
                conn_spec={'rule': 'one_to_one'}
            )
        )

for tier in range(1, 7):
    for pool in range(2):
        l2_connections_list.append(
            dict(
                pre='Tier{}E2'.format(tier),
                post='Pool{}'.format(pool),
                syn_spec=connection_params['Tier{}'.format(tier)]['e2_pool'],
                conn_spec={'rule': 'one_to_one'}
            )
        )
for pool in range(2):
    l2_connections_list.append(
        dict(
            pre='Tier0E1',
            post='Pool{}'.format(pool),
            syn_spec=connection_params['Tier0']['e1_pool{}'.format(pool)],
            conn_spec={'rule': 'one_to_one'}
        )
    )
l2_connections_list.append(
    dict(
        pre='Mediator',
        post='Tier1E0',
        syn_spec=connection_params['Mediator'],
        conn_spec={'rule': 'all_to_all'}
    )
)

""" CONNECT LEVELS """

l2_connections_list.append(
    dict(
        pre='Pool0',
        post=Layer1NeuronGroups.EXTENS_MOTOR,
        syn_spec=connection_params['Pool0']['Mn-E'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
l2_connections_list.append(
    dict(
        pre='Pool0',
        post=Layer1NeuronGroups.EXTENS_INTER_1A,
        syn_spec=connection_params['Pool0']['Ia-E'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
l2_connections_list.append(
    dict(
        pre='Pool1',
        post=Layer1NeuronGroups.FLEX_MOTOR,
        syn_spec=connection_params['Pool1']['Mn-F'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
l2_connections_list.append(
    dict(
        pre='Pool1',
        post=Layer1NeuronGroups.FLEX_INTER_1A,
        syn_spec=connection_params['Pool1']['Ia-F'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)

""" MULTIMETERS CONNECTION """
for tier in range(1, 7):
    for exc in range(5):
        l2_connections_list.append(
            dict(
                pre='Tier{}E{}-multimeter'.format(tier, exc),
                post='Tier{}E{}'.format(tier, exc)
            )
        )
    l2_connections_list.append(
        dict(
            pre='Tier{}I0-multimeter'.format(tier),
            post='Tier{}I0'.format(tier)
        )
    )
    l2_connections_list.append(
        dict(
            pre='Tier{}I1-multimeter'.format(tier),
            post='Tier{}I1'.format(tier)
        )
    )
for exc in range(2):
    l2_connections_list.append(
        dict(
            pre='Tier0E{}-multimeter'.format(exc),
            post='Tier0E{}'.format(exc)
        )
    )
l2_connections_list.append(
    dict(
        pre='Tier0I0-multimeter',
        post='Tier0I0'
    )
)
for i in range(2):
    l2_connections_list.append(
        dict(
            pre='Pool{}-multimeter'.format(i),
            post='Pool{}'.format(i)
        )
    )
l2_connections_list.append(
    dict(
        pre='spike_generator',
        post='Mediator'
    )
)