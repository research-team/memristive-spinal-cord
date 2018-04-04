import nest
import os
from the_second_level.src.tools.multimeter import add_multimeter
from the_second_level.src.tools.spike_detector import add_spike_detector
from the_second_level.src.paths import raw_data_path
from random import uniform


class Tier:

    nrn_params = {
        'V_m': -70.,
        'E_L': -70.,
        't_ref': 1.,
        'tau_syn_ex': 0.2,
        'tau_syn_in': 0.3}

    def __init__(self, index: int):
        self.e0 = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.e1 = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.e2 = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.e3 = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.e4 = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.crutch = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)

        # e0 to e1
        nest.Connect(
            pre=self.e0,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 45., 'sigma': 5.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e1 to e2
        nest.Connect(
            pre=self.e1,
            post=self.e2,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e2 to e1
        nest.Connect(
            pre=self.e2,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e0 to e3
        nest.Connect(
            pre=self.e0,
            post=self.e3,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 50., 'sigma': 5.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e3 to e4
        nest.Connect(
            pre=self.e3,
            post=self.e4,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e4 to e3
        nest.Connect(
            pre=self.e4,
            post=self.e3,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e3 to e1
        nest.Connect(
            pre=self.e3,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': -60., 'sigma': 2.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        # e4 to e1
        nest.Connect(
            pre=self.e4,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 15., 'sigma': 4.},
                'weight': {'distribution': 'normal', 'mu': -50., 'sigma': 2.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })

        # e1 to e0 [relax]
        # nest.Connect(
        #     pre=self.e1,
        #     post=self.e0,
        #     syn_spec={
        #         'model': 'static_synapse',
        #         'delay': {'distribution': 'normal', 'mu': 0.7, 'sigma': 0.1},
        #         'weight': {'distribution': 'normal', 'mu': 25., 'sigma': 2.}},
        #     conn_spec={
        #         'rule': 'all_to_all'
        #     })

        # e0 to crutch
        nest.Connect(
            pre=self.e0,
            post=self.crutch,
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.5,
                'weight': 300.
            },
            conn_spec={
                'rule': 'one_to_one'
            })

        # e0 to crutch
        nest.Connect(
            pre=self.crutch,
            post=self.e0,
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.5,
                'weight': 300.
            },
            conn_spec={
                'rule': 'one_to_one'
            })

        # mm to e0
        nest.Connect(
            pre=add_multimeter('tier{}e0'.format(index)),
            post=self.e0)
        # mm to e1
        nest.Connect(
            pre=add_multimeter('tier{}e1'.format(index)),
            post=self.e1)
        # mm to e2
        nest.Connect(
            pre=add_multimeter('tier{}e2'.format(index)),
            post=self.e2)
        # mm to e3
        nest.Connect(
            pre=add_multimeter('tier{}e3'.format(index)),
            post=self.e3)
        # mm to e4
        nest.Connect(
            pre=add_multimeter('tier{}e4'.format(index)),
            post=self.e4)

        # sd to e0
        nest.Connect(
            pre=self.e0,
            post=add_spike_detector('tier{}e0'.format(index)))
        # sd to e1
        nest.Connect(
            pre=self.e1,
            post=add_spike_detector('tier{}e1'.format(index)))
        # sd to e2
        nest.Connect(
            pre=self.e2,
            post=add_spike_detector('tier{}e2'.format(index)))
        # sd to e3
        nest.Connect(
            pre=self.e3,
            post=add_spike_detector('tier{}e3'.format(index)))
        # sd to e4
        nest.Connect(
            pre=self.e4,
            post=add_spike_detector('tier{}e4'.format(index)))

class Topology:

    def __init__(self, n_spikes: int):

        tiers = 6
        
        self.tiers = [Tier(i) for i in range(1, tiers+1)]
        self.ees = nest.Create(
            model='spike_generator',
            n=1,
            params={'spike_times': [10. + i * 35. for i in range(n_spikes)]})
        self.pool = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70., 'E_L': -70., 'tau_syn_ex': 0.5
            })
        nest.Connect(
            pre=add_multimeter('pool'),
            post=self.pool)
        self.moto = nest.Create(
            model='hh_moto_5ht',
            n=169)
        nest.Connect(
            pre=add_multimeter('moto'),
            post=self.moto)
        # EES to right_1
        nest.Connect(
            pre=self.pool,
            post=self.moto,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.7, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 25., 'sigma': 2.}},
            conn_spec={
                'rule': 'all_to_all'
            })
        nest.Connect(
            pre=self.ees,
            post=self.tiers[0].e0,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 300.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 20,
                'multapses': False
            })
        nest.Connect(
            pre=self.ees,
            post=self.moto,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 1750.
            },
            conn_spec={
                'rule': 'all_to_all'
            })

        for tier in range(tiers-1):
            # tier_1 to tier_2 [e0-e0]
            nest.Connect(
                pre=self.tiers[tier].e0,
                post=self.tiers[tier+1].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': .4,
                    'weight': 37.
                },
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 5
                })
            # tier_2 to tier_1 [e2-e2]
            nest.Connect(
                pre=self.tiers[tier+1].e2,
                post=self.tiers[tier].e2,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.7, 'sigma': 0.1},
                    'weight': {'distribution': 'normal', 'mu': 35., 'sigma': 2.}},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 5
                })
            # tier_1 to tier_2 [e3-e0]
            nest.Connect(
                pre=self.tiers[tier].e3,
                post=self.tiers[tier+1].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': .4,
                    'weight': 0.
                },
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 7
                })
        for tier in range(tiers):
            nest.Connect(
                pre=self.tiers[tier].e2,
                post=self.pool,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.7, 'sigma': 0.1},
                    'weight': {'distribution': 'normal', 'mu': 100., 'sigma': 0.2}},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 5
                })
