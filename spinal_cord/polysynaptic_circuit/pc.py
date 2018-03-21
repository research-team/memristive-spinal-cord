import nest
from spinal_cord.polysynaptic_circuit.tier import Tier
from spinal_cord.pool.pool import Pool
from spinal_cord.weights import Weights


class PolysynapticCircuit:

    def __init__(self):

        self.tiers = [
            Tier(i) for i in range(6)
        ]
        for i in range(len(self.tiers)-1):
            nest.Connect(
                pre=self.tiers[i].e[0],
                post=self.tiers[i+1].e[0],
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 1.,
                    'weight': Weights.e0e0
                },
                conn_spec={
                    'rule': 'one_to_one'
                }
            )
            nest.Connect(
                pre=self.tiers[i].e[3],
                post=self.tiers[i+1].e[0],
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 1.,
                    'weight': Weights.e3e0
                },
                conn_spec={
                    'rule': 'one_to_one'
                }
            )
            nest.Connect(
                pre=self.tiers[i+1].e[2],
                post=self.tiers[i].e[2],
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 1.,
                    'weight': Weights.e2e2
                },
                conn_spec={
                    'rule': 'one_to_one'
                }
            )

    def connect_pool(self, pool: Pool):
        for tier in self.tiers:
            for post in [pool.extens_group_nrn_ids, pool.flex_group_nrn_ids]:
                nest.Connect(
                    pre=tier.e[2],
                    post=post,
                    syn_spec={
                        'model': 'static_synapse',
                        'delay': 1.,
                        'weight': Weights.e2p
                    },
                    conn_spec={
                        'rule': 'fixed_outdegree',
                        'outdegree': 12,
                        'multapses': False
                    }
                )
        nest.Connect(
            pre=pool.extens_suspended_nrn_id,
            post=self.tiers[0].e[0],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 77.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 20,
                'multapses': False
            }
        )
