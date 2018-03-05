import nest
from spinal_cord.afferents.afferent_fiber import AfferentFiber
from spinal_cord.namespace import Muscle
from spinal_cord.toolkit.multimeter import add_multimeter
from spinal_cord.weights import Weights


class Motogroup:

    distr_normal2 = {'distribution': 'normal', 'mu': 2.0, 'sigma': 0.175}
    distr_normal3 = {'distribution': 'normal', 'mu': 3.0, 'sigma': 0.175}
    number_of_interneurons = 196
    interneuron_model = 'hh_cond_exp_traub'
    int_params = {
        't_ref': 2.,  # Refractory period
        'V_m': -70.0,  #
        'E_L': -70.0,  #
        'E_K': -77.0,  #
        'g_L': 30.0,  #
        'g_Na': 12000.0,  #
        'g_K': 3600.0,  #
        'C_m': 134.0,  # Capacity of membrane (pF)
        'tau_syn_ex': 0.5,  # Time of excitatory action (ms)
        'tau_syn_in': 5.0  # Time of inhibitory action (ms)
    }

    def __init__(self, muscle: Muscle):
        self.motoname = 'moto_{}'.format(muscle.value)
        self.ia_name = 'ia_{}'.format(muscle.value)
        self.ii_name = 'ii_{}'.format(muscle.value)
        self.moto_ids = nest.Create(
            model='hh_moto_5ht',
            n=169,
            params={
                'tau_syn_ex': 0.5,
                'tau_syn_in': 1.5,
                't_ref': 2.,  # 'tau_m': 2.5
            }
        )

        self.ia_ids = nest.Create(
            model='hh_cond_exp_traub',
            params=self.int_params,
            n=self.number_of_interneurons
        )
        self.ii_ids = nest.Create(
            model=self.interneuron_model,
            params=self.int_params,
            n=self.number_of_interneurons
        )
        nest.Connect(
            pre=self.ii_ids,
            post=self.moto_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': self.distr_normal2,
                'weight': Weights.ii_moto.value
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 116
            }
        )
        nest.Connect(
            pre=add_multimeter(self.motoname),
            post=self.moto_ids
        )
        nest.Connect(
            pre=add_multimeter(self.ia_name),
            post=self.ia_ids
        )
        nest.Connect(
            pre=add_multimeter(self.ii_name),
            post=self.ii_ids
        )

    def connect_motogroup(self, motogroup):
        if motogroup.motoname == 'moto_GM':
            nest.Connect(
                pre=self.ia_ids,
                post=motogroup.ia_ids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': self.distr_normal2,
                    'weight': 0
                },
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 100
                }
            )
        else:
            nest.Connect(
                pre=self.ia_ids,
                post=motogroup.ia_ids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': self.distr_normal2,
                    'weight': Weights.ia_ia.value
                },
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 100
                }
            )
        nest.Connect(
            pre=self.ia_ids,
            post=motogroup.moto_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.ia_moto.value
            }
        )

    def connect_afferents(self, afferent_ia: AfferentFiber, afferent_ii: AfferentFiber):
        nest.Connect(
            pre=afferent_ia.neuron_ids,
            post=self.moto_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': self.distr_normal2,
                'weight': Weights.aff_ia_moto.value
            },
            conn_spec={
                'rule': 'all_to_all',
            }
        )
        nest.Connect(
            pre=afferent_ia.neuron_ids,
            post=self.ia_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': self.distr_normal2,
                'weight': Weights.aff_ia_ia.value
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 62
            }
        )
        nest.Connect(
            pre=afferent_ii.neuron_ids,
            post=self.ia_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': self.distr_normal2,
                'weight': Weights.aff_ii_ia.value
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 62
            }
        )
        nest.Connect(
            pre=afferent_ii.neuron_ids,
            post=self.ii_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': self.distr_normal3,
                'weight': Weights.aff_ii_ii.value
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 62
            }
        )
