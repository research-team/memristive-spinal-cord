from nest import Create, Connect
from ows.src.tools.multimeter import add_multimeter
from ows.src.tools.spike_detector import add_spike_detector
from ows.src.params import inh_coef


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