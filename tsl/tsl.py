import os
from tsl_simple.src.tools.multimeter import add_multimeter
from tsl_simple.src.tools.spike_detector import add_spike_detector
from tsl_simple.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from tsl_simple.src.params import num_sublevels, num_spikes
from random import uniform


class Motoneuron:

    def create(n):
        Create(
            model='hh_cond_exp_traub',
            n=n,
            params={
                'C_m': 500.,
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'g_L': 75.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 1.0})
        Motoneuron = create(169)
        Connect(
            pre=add_multimeter('moto'),
            post=Motoneuron)
        Connect(
            pre=Motoneuron,
            post=add_spike_detector('moto'))


class Pool_Sublevel:

    def create(n):
        Create(
            model='hh_cond_exp_traub',
            n=n,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'g_L': 75.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        Pool = create(40)
        Right = create(20)
        Left = create(20)

        Connect(pre=add_multimeter('pool'),
                post=Pool)
        Connect(pre=Pool,
                post=add_spike_detector('pool'))
        Connect(pre=add_multimeter('{}{}'.format(name_right, index)),
                post=Right)
        Connect(pre=add_multimeter('{}{}'.format(name_left, index)),
                post=Left)
        Connect(pre=Right,
                post=add_spike_detector('{}{}'.format(name_right, index))),
        Connect(pre=Left,
                post=add_spike_detector('{}{}'.format(name_left, index)))


class EES:

    def create(n):
        Create(
            model='spike_generator',
            n=1,
            params={'spike_times': [25. + i * 25. for i in range(num_spikes)]})
        EES = create(1)


class Connections:

    def connect(self, pre, post, weight, degree):
            Connect(
                pre=pre,
                post=post,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': weight
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': degree})

            connect((Right, Left, 0, 4),
                    (EES, Right[0], 75, 50),
                    (EES, Moto, 1750, 169),
                    (Right[i], Right[i + 1], 15, 20),
                    (Left[i + 1], Left[i], 15, 20),
                    (Left[i], Pool, 15, 40),
                    (Pool, Moto, 15, 169))
