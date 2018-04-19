import os
from monolayer.src.tools.multimeter import add_multimeter
from monolayer.src.tools.spike_detector import add_spike_detector
from monolayer.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from monolayer.src.params import num_spikes

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
                    'weight': 0.
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
                    'weight': 0.
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
        self.exc = Create(
            model='hh_cond_exp_traub',
            n=100,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.inh = Create(
            model='hh_cond_exp_traub',
            n=100,
            params={
                'V_m': -70.,
                'E_L': -70.})
        self.pool = Pool()
        self.moto = Motoneuron()
        self.connect_ees(num_spikes)

        Connect(
            pre=add_multimeter('exc'),
            post=self.exc)
        Connect(
            pre=add_multimeter('inh'),
            post=self.inh)
        Connect(
            pre=add_multimeter('aff'),
            post=self.inh)
        Connect(
            pre=self.exc,
            post=add_spike_detector('exc'))
        Connect(
            pre=self.inh,
            post=add_spike_detector('inh'))
        Connect(
            pre=self.inh,
            post=add_spike_detector('aff'))

        Connect(
            pre=self.exc,
            post=self.inh,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
                },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 69,
                'multapses': True,
                'autapses': True})

        Connect(
            pre=self.inh,
            post=self.exc,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
                },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 69,
                'multapses': True,
                'autapses': True})

        Connect(
            pre=self.aff,
            post=self.inh,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
                },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 69,
                'multapses': True,
                'autapses': True})

        Connect(
            pre=self.aff,
            post=self.exc,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
                },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 69,
                'multapses': True,
                'autapses': True})

        Connect(
            pre=self.exc,
            post=self.pool.gids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
                },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 73,
                'multapses': True,
                'autapses': True})

