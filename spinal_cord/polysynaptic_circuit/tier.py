import nest


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
        'tau_syn_ex': 0.5,  # Time of excitatory action (ms)
        'tau_syn_in': 5.0  # Time of inhibitory action (ms)
    }

    def __init__(self):
        self.e = [
            nest.Create(
                model='hh_cond_exp_traub',
                n=20,
                params=self.params
            ) for _ in range(5)
        ]
        self.i = [
            nest.Create(
                model='hh_cond_exp_traub',
                n=20,
                params=self.params
            ) for _ in range(2)
        ]
        nest.Connect(
            pre=self.e[0],
            post=self.e[1],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 100.
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
                'delay': 1.,
                'weight': 100.
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
                'delay': 1.,
                'weight': 100.
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
                'delay': 1.,
                'weight': 100.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.e[4],
            post=self.e[3],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 100.
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
                'delay': 1.,
                'weight': 100.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.e[3],
            post=self.i[0],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 100.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.i[0],
            post=self.e[1],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': -25.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.i[1],
            post=self.e[1],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': -25.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.e[2],
            post=self.i[1],
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 100.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )