from memristive_spinal_cord.proposed_scheme.level2.parameters import Weights


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
    'e0_e1': {
        'model': 'static_synapse',
        'delay': distr_normal_2,
        'weight': Weights.EE.value[0][0]
    },
    'e1_e0': {
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