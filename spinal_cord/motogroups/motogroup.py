import nest


class Motogroup:

    distr_normal2 = {'distribution': 'normal', 'mu': 2.0, 'sigma': 0.175}
    distr_normal_3 = {'distribution': 'normal', 'mu': 3.0, 'sigma': 0.175}
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

    def __init__(self):

        self.moto_ids = nest.Create(
            model='hh_moto_5ht',
            n=169,
            params={
                'tau_syn_ex': 0.5,
                'tau_syn_in': 1.5,
                't_ref': 2.0,  # 'tau_m': 2.5
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
                'weight': 15.419
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 116
            }
        )

    def connect(self, motogroup):
        nest.Connect(
            pre=self.ia_ids,
            post=motogroup.ia_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': self.distr_normal2,
                'weight': -0.7
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 100
            }
        )
