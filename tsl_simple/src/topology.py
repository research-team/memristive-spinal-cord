import os
from tsl_simple.src.tools.multimeter import add_multimeter
from tsl_simple.src.tools.spike_detector import add_spike_detector
from tsl_simple.src.paths import raw_data_path
from nest import Create, Connect


class Topology:
    nrn_params = {
        'V_m': -70.,
        'E_L': -70.,
        't_ref': 1.,
        'tau_syn_ex': 0.2,
        'tau_syn_in': 0.3}

    def __init__(self, num_spikes: int):
        self.right_1 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.left_1 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.hright_1 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.hleft_1 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.right_2 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)
        self.left_2 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.nrn_params)

        self.ees = Create(
            model='spike_generator',
            n=1,
            params={'spike_times': [10. + i * 25. for i in range(num_spikes)]})

        Connect(
            pre=self.ees,
            post=self.right_1,
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
        Connect(
            pre=self.right_1,
            post=self.left_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.left_1,
            post=self.right_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.hright_1,
            post=self.hleft_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.hleft_1,
            post=self.hright_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.right_2,
            post=self.left_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 5
            })
        Connect(
            pre=self.left_2,
            post=self.right_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.right_1,
            post=self.right_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 25., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.right_1,
            post=self.hright_1,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })
        Connect(
            pre=self.hright_1,
            post=self.right_2,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 25., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5
            })

        # mm to right_1
        Connect(
            pre=add_multimeter('right_1'),
            post=self.right_1)
        # mm to left_1
        Connect(
            pre=add_multimeter('left_1'),
            post=self.left_1)
        # mm to hright_1
        Connect(
            pre=add_multimeter('hight_1'),
            post=self.hright_1)
        # mm to hleft_1
        Connect(
            pre=add_multimeter('heft_1'),
            post=self.hleft_1)
        # mm to right_2
        Connect(
            pre=add_multimeter('right_2'),
            post=self.right_2)
        # mm to left_2
        Connect(
            pre=add_multimeter('left_2'),
            post=self.left_2)

        # sd to right_1
        Connect(
            pre=self.right_1,
            post=add_spike_detector('right_1'))
        # sd to left_1
        Connect(
            pre=self.left_1,
            post=add_spike_detector('left_1'))
        # sd to right_1
        Connect(
            pre=self.hright_1,
            post=add_spike_detector('hight_1'))
        # sd to left_1
        Connect(
            pre=self.hleft_1,
            post=add_spike_detector('heft_1'))
        # sd to right_2
        Connect(
            pre=self.right_2,
            post=add_spike_detector('right_2'))
        # sd to left_2
        Connect(
            pre=self.left_2,
            post=add_spike_detector('left_2'))
