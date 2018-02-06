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
            'model': 'static_synase',
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
        'e0_e4': {
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
    if tier > 1:
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
        'weight': -Weights.IE.value[1]
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
        'weight': Weights.PE.value
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

""" INTERCONNECTIONS """
connections_list = []

for tier in range(1, 7):
    connections_list.append(
        dict(
            pre='Tier{}E0'.format(tier),
            post='Tier{}E1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e0_e1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}E1'.format(tier),
            post='Tier{}E2'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e1_e2'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}E2'.format(tier),
            post='Tier{}E1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e2_e1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}E3'.format(tier),
            post='Tier{}E4'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e3_e4'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}E4'.format(tier),
            post='Tier{}E3'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e4_e3'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}E0'.format(tier),
            post='Tier{}E3'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e0_e3'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}E3'.format(tier),
            post='Tier{}I0'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['e3_i0'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    connections_list.append(
        dict(
            pre='Tier{}I0'.format(tier),
            post='Tier{}E1'.format(tier),
            syn_spec=connection_params['Tier{}'.format(tier)]['i0_e1'],
            conn_spec={'rule': 'one_to_one'}
        )
    )
    if tier < 6:
        connections_list.append(
            dict(
                pre='Tier{}E0'.format(tier),
                post='Tier{}E0'.format(tier + 1),
                syn_spec=connection_params['Tier{}'.format(tier)]['e0_e0'],
                conn_spec={'rule': 'one_to_one'}
            )
        )
        connections_list.append(
            dict(
                pre='Tier{}E3'.format(tier),
                post='Tier{}E0'.format(tier + 1),
                syn_spec=connection_params['Tier{}'.format(tier)]['e3_e0'],
                conn_spec={'rule': 'one_to_one'}
            )
        )
    if tier > 1:
        connections_list.append(
            dict(
                pre='Tier{}E2'.format(tier),
                post='Tier{}E2'.format(tier + 1),
                syn_spec=connection_params['Tier{}'.format(tier)]['e2_e2'],
                conn_spec={'rule': 'one_to_one'}
            )
        )

for tier in range(1, 7):
    for pool in range(2):
        connections_list.append(
            dict(
                pre='Tier{}'.format(tier),
                post='Pool{}'.format(pool),
                syn_spec=connection_params['Tier{}'.format(tier)]['e2_pool{}'.format(pool)],
                conn_spec={'rule': 'one_to_one'}
            )
        )
for pool in range(2):
    connections_list.append(
        dict(
            pre='Tier0E1',
            post='Pool{}'.format(pool),
            syn_spec=connection_params['Tier0']['e1_pool{}'.format(pool)],
            conn_spec={'rule': 'one_to_one'}
        )
    )

""" CONNECT LEVELS """

connections_list.append(
    dict(
        pre='Pool0',
        post=Layer1NeuronGroups.EXTENS_MOTOR,
        syn_spec=connection_params['Pool0']['Mn-E'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
connections_list.append(
    dict(
        pre='Pool0',
        post=Layer1NeuronGroups.EXTENS_INTER_1A,
        syn_spec=connection_params['Pool0']['Ia-E'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
connections_list.append(
    dict(
        pre='Pool1',
        post=Layer1NeuronGroups.FLEX_MOTOR,
        syn_spec=connection_params['Pool1']['Mn-F'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
connections_list.append(
    dict(
        pre='Pool1',
        post=Layer1NeuronGroups.FLEX_INTER_1A,
        syn_spec=connection_params['Pool1']['Ia-F'],
        conn_spec={'rule': 'fixed_indegree', 'indegree': 100}
    )
)
