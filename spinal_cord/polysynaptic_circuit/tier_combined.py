import nest
import nest.visualization
from spinal_cord.params import Params
from spinal_cord.toolkit.multimeter import add_multimeter
from spinal_cord.weights import Weights
from pkg_resources import resource_filename


class Tier:
    params = {
        't_ref': 2.,  # Refractory period
        'V_m': -70.0,  #
        'E_L': -70.0,  #
        'E_K': -77.0,  #
        'g_L': 30.0,  #
        'g_Na': 12000.0,  #
        'g_K': 3600.0,  #
        'C_m': 134.0,  # Capacity of membrane (pF)
        'tau_syn_ex': 4.7,  # Time of excitatory action (ms)
        'tau_syn_in': 3.1  # Time of inhibitory action (ms)
    }

    def __init__(self, index: int):
        self.index = index
        self.e = []
        for _ in range(5):
            self.e.append(
                nest.Create(
                    model='hh_cond_exp_traub',
                    n=20,
                    params=self.params
                )
            )
        self.i = []
        for _ in range(2):
            self.i.append(
                nest.Create(
                    model='hh_cond_exp_traub',
                    n=20,
                    params=self.params
                )
            )
        for ex in self.e:
            Weights.ids.append(ex[0])

        nest.Connect(
            pre=self.e[0],
            post=self.e[1],
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.9,
                'weight': Weights.e0e1
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.e[1],
            post=self.e[2],
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.9,
                'weight': Weights.e1e2
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.e[2],
            post=self.e[1],
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.9,
                'weight': Weights.e2e1
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.e[3],
            post=self.e[4],
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.9,
                'weight': Weights.e3e4
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        a = nest.Create("weight_recorder", 1, {'to_memory': False, 'to_file': True, 'label': '{}/{}/{}'.format(resource_filename('spinal_cord', 'results'), 'raw_data', 'kek')})
        name = 'wcr_syn{}'.format(self.index)
        nest.CopyModel('stdp_synapse', name, {'weight_recorder': a[0]})
        nest.Connect(
            pre=self.e[4],
            post=self.e[3],
            syn_spec={
                'model': name,
                'delay': 0.9,
                'alpha': 1.0,  # Coeficient for inhibitory STDP time (alpha * lambda)
                'lambda': 0.0037,  # Time interval for STDP
                'Wmax': 100.,  # Maximum possible weight
                'mu_minus': 0.,  # STDP depression step
                'mu_plus': 0.,  # STDP potential step
                'weight': Weights.e4e3
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )

        nest.Connect(
            pre=self.e[0],
            post=self.e[3],
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.9,
                'weight': Weights.e0e3
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        # nest.Connect(
        #     pre=self.e[3],
        #     post=self.i[0],
        #     syn_spec={
        #         'model': 'static_synapse',
        #         'delay': 0.9,
        #         'weight': Weights.e3i0
        #     },
        #     conn_spec={
        #         'rule': 'one_to_one'
        #     }
        # )
        # nest.Connect(
        #     pre=self.i[0],
        #     post=self.e[1],
        #     syn_spec={
        #         'model': 'static_synapse',
        #         'delay': 0.9,
        #         'weight': Weights.i0e1
        #     },
        #     conn_spec={
        #         'rule': 'one_to_one'
        #     }
        # )
        # nest.Connect(
        #     pre=self.i[1],
        #     post=self.e[1],
        #     syn_spec={
        #         'model': 'static_synapse',
        #         'delay': 0.9,
        #         'weight': Weights.i1e1
        #     },
        #     conn_spec={
        #         'rule': 'one_to_one'
        #     }
        # )
        # nest.Connect(
        #     pre=self.e[2],
        #     post=self.i[1],
        #     syn_spec={
        #         'model': 'static_synapse',
        #         'delay': 0.9,
        #         'weight': Weights.e2i1
        #     },
        #     conn_spec={
        #         'rule': 'one_to_one'
        #     }
        # )
        b = nest.Create("weight_recorder", 1, {'to_memory': False, 'to_file': True,
                                               'label': '{}/{}/{}'.format(resource_filename('spinal_cord', 'results'),
                                                                          'raw_data', 'e3e1')})
        name = 'e3e1_syn{}'.format(self.index)
        nest.CopyModel('static_synapse', name, {'weight_recorder': a[0]})
        nest.Connect(
            pre=self.e[3],
            post=self.e[1],
            syn_spec={
                'model': name,
                'delay': 0.9,
                'weight': Weights.e3i0e1
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.visualization.plot_network(Weights.ids, filename='/home/cmen/lol0.pdf', ext_conns=False)
        for i in range(len(self.e)):
            nest.Connect(
                pre=add_multimeter('tier{}e{}'.format(self.index, i)),
                post=self.e[i]
            )
        for i in range(len(self.i)):
            nest.Connect(
                pre=add_multimeter('tier{}i{}'.format(self.index, i)),
                post=self.i[i]
            )