import nest
from spinal_cord.namespace import Muscle, Afferent


class AfferentFiber:

    def __init__(self, muscle: Muscle, afferent: Afferent):
        self.muscle = muscle
        self.afferent = afferent
        self.neuron_ids = nest.Create(
            model='iaf_psc_alpha',
            n=60,
            params={
                'V_m': -70.0,
                'V_reset': -65.0,
                'V_th': -55.0,
                'tau_m': 0.5,
                'tau_syn_ex': 0.2,
                't_ref': 1.0,
            }
        )
