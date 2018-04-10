import os
from tsl_simple.src.tools.multimeter import add_multimeter
from tsl_simple.src.tools.spike_detector import add_spike_detector
from tsl_simple.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from tsl_simple.src.params import num_sublevels, num_spikes, Connections, Neurons
from random import uniform


class Motoneuron:

    def __init__(self):
        self.gids = Create(
            model='hh_cond_exp_traub',
            n=Neurons.moto,
            params={
                'C_m': 500.,
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
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
            n=Neurons.pool,
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
            n=Neurons.sublayers['sublayer_{}'.format(index)]['right'],
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        self.left = Create(
            model='hh_cond_exp_traub',
            n=Neurons.sublayers['sublayer_{}'.format(index)]['left'],
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})

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
                'delay': 0.4,
                'weight': Connections.sub_right_to_left['sublayer_{}'.format(index)]['weight']},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': Connections.sub_right_to_left['sublayer_{}'.format(index)]['degree']})
        Connect(
            pre=self.left,
            post=self.right,
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.4,
                'weight': Connections.sub_left_to_right['sublayer_{}'.format(index)]['weight']},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': Connections.sub_left_to_right['sublayer_{}'.format(index)]['degree']})

class HiddenSublevel:

    def __init__(self, index: int, name_left: str='left', name_right: str='right'):
        self.right = Create(
            model='hh_cond_exp_traub',
            n=Neurons.hidden_sublayers['sublayer_{}'.format(index)]['right'],
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        self.left = Create(
            model='hh_cond_exp_traub',
            n=Neurons.hidden_sublayers['sublayer_{}'.format(index)]['left'],
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})

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
                'delay': 0.4,
                'weight': Connections.hidden_sub_right_to_left['sublayer_{}'.format(index)]['weight']},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': Connections.hidden_sub_right_to_left['sublayer_{}'.format(index)]['degree']})
        Connect(
            pre=self.left,
            post=self.right,
            syn_spec={
                'model': 'static_synapse',
                'delay': 0.4,
                'weight': Connections.hidden_sub_left_to_right['sublayer_{}'.format(index)]['weight']},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': Connections.hidden_sub_left_to_right['sublayer_{}'.format(index)]['degree']})  


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
                    'weight': 25.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 2000})
            Connect(
                pre=self.ees,
                post=self.moto.gids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 1000.
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
                pre=self.sublevels[i].right,
                post=self.sublevels[i+1].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': .4,
                    'weight': Connections.right_to_right_up[
                        'sublayers_{}_{}'.format(i, i + 1)]['weight']},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': Connections.right_to_right_up[
                        'sublayers_{}_{}'.format(i, i + 1)]['degree']})
            Connect(
                pre=self.hidden_sublevels[i].right,
                post=self.sublevels[i+1].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': Connections.hidden_right_to_right_up[
                        'sublayer_{}_{}'.format(i, i+1)]['weight']},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': Connections.hidden_right_to_right_up[
                        'sublayer_{}_{}'.format(i, i+1)]['degree']})
        for i in range(num_sublevels):
            Connect(
                pre=self.sublevels[i].right,
                post=self.hidden_sublevels[i].right,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': Connections.right_to_hidden_right_up[
                        'sublayer_{}'.format(i)]['weight']},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': Connections.right_to_hidden_right_up[
                        'sublayer_{}'.format(i)]['degree']})
            Connect(
                pre=self.hidden_sublevels[i].left,
                post=self.sublevels[i].left,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': Connections.hidden_left_to_left_down[
                        'sublayer_{}'.format(i)]['weight']},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': Connections.hidden_left_to_left_down[
                        'sublayer_{}'.format(i)]['degree']})


        # pool to moto

        for i in range(num_sublevels):
            Connect(
                pre=self.sublevels[i].left,
                post=self.pool.gids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                    'weight': Connections.left_to_pool[
                        'sublayer_{}'.format(i)]['weight']},
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': Connections.left_to_pool[
                        'sublayer_{}'.format(i)]['degree']
                })

        Connect(
            pre=self.pool.gids,
            post=self.moto.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': {'distribution': 'normal', 'mu': 0.5, 'sigma': 0.1},
                'weight': Connections.pool_to_moto['weight']},
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': Connections.pool_to_moto['degree']})
