import nest


class Pool:

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
        self.suspended_flex_nrn_id = nest.Create(
            model='hh_cond_exp_traub',
            n=1,
            params=self.params
        )
        self.suspended_extens_nrn_id = nest.Create(
            model='hh_cond_exp_traub',
            n=1,
            params=self.params
        )
        self.flex_group_nrn_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.params
        )
        self.extens_group_nrn_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.params
        )
        nest.Connect(
            pre=self.suspended_extens_nrn_id,
            post=self.extens_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 100.
            },
            conn_spec={
                'rule': 'all_to_all'
            }
        )
        nest.Connect(
            pre=self.suspended_flex_nrn_id,
            post=self.flex_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 100.
            },
            conn_spec={
                'rule': 'all_to_all'
            }
        )
