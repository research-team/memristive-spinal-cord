import os
from lcr_complex.src.tools.multimeter import add_multimeter
from lcr_complex.src.tools.spike_detector import add_spike_detector
from lcr_complex.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from lcr_complex.src.params import num_sublevels, num_spikes
from random import uniform


class Motoneuron:

    def __init__(self):
        self.gids = Create(
            model='hh_cond_exp_traub',
            n=169,
            params={
                'C_m': 500.,
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 1.0})
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
            n=40,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        Connect(
            pre=add_multimeter('pool'),
            post=self.gids)
        Connect(
            pre=self.gids,
            post=add_spike_detector('pool'))


class Sublevel:

    def __init__(self, index: int, name_left: str='left', name_right: str='right'):
        self.right = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': .5})
        self.left = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': .5})

        Connect(
            pre=add_multimeter('{}{}'.format(name_right, index)),
            post=self.right)
        Connect(
            pre=add_multimeter('{}{}'.format(name_left, index)),
            post=self.left)
        Connect(
            pre=self.right,
            post=add_spike_detector('{}{}'.format(name_right, index)))
        Connect(
            pre=self.left,
            post=add_spike_detector('{}{}'.format(name_left, index)))

        Connect(
            pre=self.right,
            post=self.left,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 2.
                },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 3})

class HiddenSublevel:

    def __init__(self, index: int, name_left: str='left', name_right: str='right'):
        self.right = Create(
            model='hh_cond_exp_traub',
            n=10,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': .5})
        self.left = Create(
            model='hh_cond_exp_traub',
            n=10,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': .5})

        Connect(
            pre=add_multimeter('{}{}'.format(name_right, index)),
            post=self.right)
        Connect(
            pre=add_multimeter('{}{}'.format(name_left, index)),
            post=self.left)
        Connect(
            pre=self.right,
            post=add_spike_detector('{}{}'.format(name_right, index)))
        Connect(
            pre=self.left,
            post=add_spike_detector('{}{}'.format(name_left, index)))

        Connect(
            pre=self.right,
            post=self.left,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 0.},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5})
        Connect(
            pre=self.left,
            post=self.right,
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.1,
                'weight': 0.},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5})  


class Topology:

    def connect_ees(self, num_spikes):

        if num_spikes:
            self.ees = Create(
                model='spike_generator',
                n=1,
                params={'spike_times': [25. + i * 25. for i in range(num_spikes)]})
            Connect(
                pre=self.ees,
                post=self.sublevels[0].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 200.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 10,
                    'multapses': False})
            Connect(
                pre=self.ees,
                post=self.moto.gids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 1750.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 169,
                    'multapses': False})

    def __init__(self):

        self.sublevels = [Sublevel(index=i) for i in range(num_sublevels)]
        self.hidden_sublevels = [HiddenSublevel(
            index=i,
            name_left='heft',
            name_right='hight') for i in range(num_sublevels)]
        self.pool = Pool()
        self.moto = Motoneuron()
        self.connect_ees(num_spikes)

        for i in range(num_sublevels-1):
            Connect(
                pre=self.hidden_sublevels[i].left,
                post=self.sublevels[i+1].left,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 0.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})
            Connect(
                pre=self.sublevels[i].right,
                post=self.sublevels[i+1].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 0.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 1})
            Connect(
                pre=self.hidden_sublevels[i].right,
                post=self.sublevels[i+1].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 0.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 1})
            Connect(
                pre=self.sublevels[i].left,
                post=self.sublevels[i+1].left,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 0.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})
        for i in range(num_sublevels):
            Connect(
                pre=self.sublevels[i].right,
                post=self.hidden_sublevels[i].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 0.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})
            Connect(
                pre=self.hidden_sublevels[i].left,
                post=self.sublevels[i].left,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 0},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': 1})


        # pool to moto

        for i in range(num_sublevels):
            Connect(
                pre=self.sublevels[i].left,
                post=self.pool.gids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 150.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5
                })

        Connect(
            pre=self.pool.gids,
            post=self.moto.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.1,
                'weight': 500.},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 7})
