import os
from two_sublayers.src.tools.multimeter import add_multimeter
from two_sublayers.src.tools.spike_detector import add_spike_detector
from two_sublayers.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from two_sublayers.src.params import num_spikes

class Motoneuron:

    def __init__(self):
        self.gids = Create(
            model='hh_cond_exp_traub',
            n=169,
            params={
                'C_m': 500.,
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.})
        Connect(
            pre=add_multimeter('moto'),
            post=self.gids)
        Connect(
            pre=self.gids,
            post=add_spike_detector('moto'))


class Pool:
    def __init__(self):
        self.gids = Create(
            model='hh_cond_exp_traub',
            n=120,
            params={
                'V_m': -70.,
                'E_L': -70.})
        Connect(
            pre=add_multimeter('pool'),
            post=self.gids)
        Connect(
            pre=self.gids,
            post=add_spike_detector('pool'))

class Topology:

    def connect_ees(self, num_spikes):

        if num_spikes:
            self.ees = Create(
                model='spike_generator',
                n=1,
                params={'spike_times': [25. + i * 25. for i in range(num_spikes)]})
            Connect(
                pre=self.ees,
                post=self.aff,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 20.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 60,
                    'multapses': False})
            Connect(
                pre=self.ees,
                post=self.moto.gids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 50.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 169,
                    'multapses': False})

    def __init__(self):

        self.aff = Create(
            model='hh_cond_exp_traub',
            n=60,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.right_1 = Create(
            model='hh_cond_exp_traub',
            n=50,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.right_2 = Create(
            model='hh_cond_exp_traub',
            n=50,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.left_1 = Create(
            model='hh_cond_exp_traub',
            n=50,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.left_2 = Create(
            model='hh_cond_exp_traub',
            n=50,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.inh = Create(
            model='hh_cond_exp_traub',
            n=50,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.pool = Pool()
        self.moto = Motoneuron()
        self.connect_ees(num_spikes)

        Connect(
            pre=add_multimeter('right_1'),
            post=self.right_1)
        Connect(
            pre=add_multimeter('right_2'),
            post=self.right_2)
        Connect(
            pre=add_multimeter('left_1'),
            post=self.left_1)
        Connect(
            pre=add_multimeter('left_2'),
            post=self.left_2)

        Connect(
            pre=add_multimeter('inh'),
            post=self.inh)
        Connect(
            pre=add_multimeter('aff'),
            post=self.aff)

        Connect(
            pre=self.right_1,
            post=add_spike_detector('right_1'))
        Connect(
            pre=self.right_2,
            post=add_spike_detector('right_2'))
        Connect(
            pre=self.left_1,
            post=add_spike_detector('left_1'))
        Connect(
            pre=self.left_2,
            post=add_spike_detector('left_2'))

        Connect(
            pre=self.inh,
            post=add_spike_detector('inh'))
        Connect(
            pre=self.aff,
            post=add_spike_detector('aff'))

        Connect(
            pre=self.aff,
            post=self.right_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 1.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.right_1,
            post=self.right_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.5
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.right_1,
            post=self.left_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.left_1,
            post=self.left_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.right_2,
            post=self.left_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.right_1,
            post=self.inh,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.right_2,
            post=self.inh,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.left_1,
            post=self.inh,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.left_2,
            post=self.inh,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.inh,
            post=self.right_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.inh,
            post=self.right_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.inh,
            post=self.left_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.inh,
            post=self.left_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.left_1,
            post=self.pool.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.left_2,
            post=self.pool.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })

        Connect(
            pre=self.pool.gids,
            post=self.moto.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 16,
                'multapses': True,
                'autapses': True
            })