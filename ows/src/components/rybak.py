from ows.src.tools.multimeter import add_multimeter
from ows.src.tools.spike_detector import add_spike_detector
from nest import Create, Connect


def connect(pre, post, weight, num_synapses=500):
        Connect(
            pre=pre,
            post=post,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': weight
            },
            conn_spec={
                'rule': 'fixed_total_number',
                'N': num_synapses
            })

class Motoneuron:
    def __init__(self, name: str):
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
            pre=add_multimeter(name),
            post=self.gids)
        Connect(
            pre=self.gids,
            post=add_spike_detector(name))

class Interneuron:
    def __init__(self, name: str, n: int):
        self.gids = Create(
            model='hh_cond_exp_traub',
            n=n,
            params={
                'V_m': -70.,
                'E_L': -70.,
                't_ref': 1.,
                'tau_syn_ex': 0.2,
                'tau_syn_in': 0.3})
        Connect(
            pre=add_multimeter(name),
            post=self.gids)
        Connect(
            pre=self.gids,
            post=add_spike_detector(name))


class Rybak:
    def __init__(self):
        self.pool_flexor = Interneuron('flexor_pool', 120)
        self.pool_extensor = Interneuron('extensor_pool', 120)

        self.moto_flexor = Motoneuron('flexor_moto')
        self.moto_extensor = Motoneuron('extensor_moto')

        self.ib_flexor = Interneuron('flexor_ib', 196)
        self.ib_extensor = Interneuron('extensor_ib', 196)

        self.ia_flexor = Interneuron('flexor_ia', 196)
        self.ia_extensor = Interneuron('extensor_ia', 196)

        self.rc_flexor = Interneuron('rc_flexor', 196)
        self.rc_extensor = Interneuron('rc_extensor', 196)

        self.aff_ia_flexor = Interneuron('flexor_ia_afferents', 120)
        self.aff_ia_extensor = Interneuron('extensor_ia_afferents', 120)

        self.sensory_flexor = Interneuron('flexor_sensory', 120)
        self.sensory_extensor = Interneuron('extensor_sensory', 120)

        self.s0 = Interneuron('s0', 100)
        self.s1 = Interneuron('s1', 100)

        self.set_connections()

    def set_connections(self):
        connect(pre=self.sensory_flexor.gids, post=self.s0.gids, weight=120)
        connect(pre=self.sensory_extensor.gids, post=self.s1.gids, weight=120)

        connect(pre=self.s0.gids, post=self.pool_flexor.gids, weight=120)
        connect(pre=self.s1.gids, post=self.pool_extensor.gids, weight=120)

        connect(pre=self.pool_flexor.gids, post=self.pool_extensor.gids, weight=-100)
        connect(pre=self.pool_flexor.gids, post=self.moto_flexor.gids, weight=50)
        connect(pre=self.pool_flexor.gids, post=self.ia_flexor.gids, weight=50)

        connect(pre=self.pool_extensor.gids, post=self.pool_flexor.gids, weight=-50)
        connect(pre=self.pool_extensor.gids, post=self.moto_extensor.gids, weight=180)
        connect(pre=self.pool_extensor.gids, post=self.ia_extensor.gids, weight=50)

        connect(pre=self.aff_ia_flexor.gids, post=self.moto_flexor.gids, weight=30)
        connect(pre=self.aff_ia_flexor.gids, post=self.ia_flexor.gids, weight=50)
        connect(pre=self.aff_ia_flexor.gids, post=self.ib_flexor.gids, weight=50)

        connect(pre=self.aff_ia_extensor.gids, post=self.moto_extensor.gids, weight=180)
        connect(pre=self.aff_ia_extensor.gids, post=self.ia_extensor.gids, weight=50)
        connect(pre=self.aff_ia_extensor.gids, post=self.ib_extensor.gids, weight=50)

        connect(pre=self.ib_flexor.gids, post=self.ib_extensor.gids, weight=-30)
        connect(pre=self.ib_extensor.gids, post=self.ib_flexor.gids, weight=-30)
        connect(pre=self.ib_flexor.gids, post=self.moto_flexor.gids, weight=-20)
        connect(pre=self.ib_extensor.gids, post=self.moto_extensor.gids, weight=-50)

        connect(pre=self.ia_flexor.gids, post=self.ia_extensor.gids, weight=-30)
        connect(pre=self.ia_extensor.gids, post=self.ia_flexor.gids, weight=-30)
        connect(pre=self.ia_flexor.gids, post=self.moto_flexor.gids, weight=-20)
        connect(pre=self.ia_extensor.gids, post=self.moto_extensor.gids, weight=-50)

        connect(pre=self.rc_flexor.gids, post=self.rc_extensor.gids, weight=-50)
        connect(pre=self.rc_flexor.gids, post=self.ia_flexor.gids, weight=-50)
        connect(pre=self.rc_flexor.gids, post=self.moto_flexor.gids, weight=-20)

        connect(pre=self.rc_extensor.gids, post=self.rc_flexor.gids, weight=-30)
        connect(pre=self.rc_extensor.gids, post=self.ia_extensor.gids, weight=-30)
        connect(pre=self.rc_extensor.gids, post=self.moto_extensor.gids, weight=-50)
