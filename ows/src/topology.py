import os
from ows.src.tools.multimeter import add_multimeter
from ows.src.tools.spike_detector import add_spike_detector
from ows.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from ows.src.params import num_sublevels, num_spikes, rate, inh_coef
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

    def __init__(self, index: int):
        self.crutch = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': 1.})
        self.e0 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': 1.})
        self.e1 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': 1.})
        self.e2 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': 1.})
        self.e3 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': 1.})
        self.e4 = Create(
            model='hh_cond_exp_traub',
            n=20,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': .5,
                'tau_syn_in': 1.})

        Connect(
            pre=add_multimeter('{}{}'.format('e0', index)),
            post=self.e0)
        Connect(
            pre=self.e0,
            post=add_spike_detector('{}{}'.format('e0', index)))

        Connect(
            pre=add_multimeter('{}{}'.format('e1', index)),
            post=self.e1)
        Connect(
            pre=self.e1,
            post=add_spike_detector('{}{}'.format('e1', index)))

        Connect(
            pre=add_multimeter('{}{}'.format('e2', index)),
            post=self.e2)
        Connect(
            pre=self.e2,
            post=add_spike_detector('{}{}'.format('e2', index)))

        Connect(
            pre=add_multimeter('{}{}'.format('e3', index)),
            post=self.e3)
        Connect(
            pre=self.e3,
            post=add_spike_detector('{}{}'.format('e3', index)))

        Connect(
            pre=add_multimeter('{}{}'.format('e4', index)),
            post=self.e4)
        Connect(
            pre=self.e4,
            post=add_spike_detector('{}{}'.format('e4', index)))


        Connect(
            pre=self.e0,
            post=self.crutch,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 150.},
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.crutch,
            post=self.e0,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 150.
                },
            conn_spec={
                'rule': 'one_to_one'})

        Connect(
            pre=self.e0,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 100.
                },
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.e1,
            post=self.e2,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 150.
                },
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.e2,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 0.
                },
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.e0,
            post=self.e3,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 120.
                },
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.e4,
            post=self.e3,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 200.
                },
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.e3,
            post=self.e4,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': 0.
                },
            conn_spec={
                'rule': 'one_to_one'})
        Connect(
            pre=self.e3,
            post=self.e1,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': -600. * inh_coef
                },
            conn_spec={
                'rule': 'one_to_one'})

class Topology:

    def connect_ees(self, num_spikes, rate):

        period = round(1000 / rate, 1)

        if num_spikes:
            self.ees = Create(
                model='spike_generator',
                n=1,
                params={'spike_times': [period + i * period for i in range(num_spikes)]})
            Connect(
                pre=self.ees,
                post=self.sublevels[0].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.1,
                    'weight': 200.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 20,
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
        self.pool = Pool()
        self.moto = Motoneuron()
        self.connect_ees(num_spikes, rate)

        for i in range(num_sublevels-1):
            Connect(
                pre=self.sublevels[i].e0,
                post=self.sublevels[i+1].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.4,
                    'weight': 75.},
                conn_spec={
                    'rule': 'one_to_one'})
            Connect(
                pre=self.sublevels[i].e3,
                post=self.sublevels[i+1].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.4,
                    'weight': 0.},
                conn_spec={
                    'rule': 'one_to_one'})
            Connect(
                pre=self.sublevels[i].e2,
                post=self.sublevels[i+1].e2,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.4,
                    'weight': 150.},
                conn_spec={
                    'rule': 'one_to_one'})

        # pool to moto

        for i in range(num_sublevels):
            Connect(
                pre=self.sublevels[i].e2,
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
                'outdegree': 10})
