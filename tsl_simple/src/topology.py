import os
from tsl_simple.src.tools.multimeter import add_multimeter
from tsl_simple.src.tools.spike_detector import add_spike_detector
from tsl_simple.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from tsl_simple.src.params import num_sublevels, num_spikes
from random import uniform


class Motoneuron:

    def __init__(self):
        self.gids = Create(
            model='hh_moto_5ht',
            n=169)
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
            n=20,
            params={
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        V_ms = [{"V_m": uniform(-50., -60.)} for _ in self.gids]
        SetStatus(self.gids, V_ms)
        Connect(
            pre=add_multimeter('pool'),
            post=self.gids)
        Connect(
            pre=self.gids,
            post=add_spike_detector('pool'))


class Sublevel:

    def __init__(self, index: int=0):
        self.right = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        self.left = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        V_ms = [{"V_m": uniform(-50., -60.)} for _ in self.left]
        SetStatus(self.left, V_ms)
        V_ms = [{"V_m": uniform(-50., -60.)} for _ in self.right]
        SetStatus(self.right, V_ms)
        Connect(
            pre=self.right,
            post=self.left,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5})
        Connect(
            pre=self.left,
            post=self.right,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 85., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5})
        Connect(
            pre=add_multimeter('right{}'.format(index)),
            post=self.right)
        Connect(
            pre=add_multimeter('left{}'.format(index)),
            post=self.left)
        Connect(
            pre=self.right,
            post=add_spike_detector('right'))
        Connect(
            pre=self.left,
            post=add_spike_detector('left'))


class Topology:

    def __init__(self):

        self.sublevels = [Sublevel(index=i) for i in range(num_sublevels)]
        self.pool = Pool()
        self.moto = Motoneuron()

        if num_spikes:
            self.ees = Create(
                model='spike_generator',
                n=1,
                params={'spike_times': [10. + i * 25. for i in range(num_spikes)]})
            Connect(
                pre=self.ees,
                post=self.sublevels[0].right)

        for i in range(num_sublevels-1):
            Connect(
                pre=self.sublevels[i].right,
                post=self.sublevels[i+1].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': {'distribution': 'normal', 'mu': 0., 'sigma': 4.}},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})
            Connect(
                pre=self.sublevels[i+1].left,
                post=self.sublevels[i].left,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': {'distribution': 'normal', 'mu': 0., 'sigma': 4.}},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})
            Connect(
                pre=self.sublevels[i+1].right,
                post=self.sublevels[i].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': {'distribution': 'normal', 'mu': 0., 'sigma': 4.}},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})

        for i in range(num_sublevels):
            Connect(
                pre=self.sublevels[i].left,
                post=self.pool.gids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': {'distribution': 'normal', 'mu': 0., 'sigma': 4.}},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 5})

        Connect(
            pre=self.pool.gids,
            post=self.moto.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': {'distribution': 'normal', 'mu': 0., 'sigma': 4.}},
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 5})
