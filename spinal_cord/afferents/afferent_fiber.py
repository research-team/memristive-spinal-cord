import nest
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.toolkit.multimeter import add_multimeter


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
        nest.Connect(
            pre=add_multimeter('afferent_{}_fiber_{}'.format(afferent.value, muscle.value)),
            post=self.neuron_ids
        )
