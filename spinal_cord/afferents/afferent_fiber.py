import nest
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.toolkit.multimeter import add_multimeter


class AfferentFiber:

    params_ = {
        'V_m': -70.0,
        'V_reset': -65.0,
        'V_th': -55.0,
        'tau_m': .5,
        'tau_syn_ex': 0.2,
        't_ref': 1.0,
    }

    def __init__(self, muscle: Muscle, afferent: Afferent):
        self.name = 'afferent_{}_fiber_{}'.format(muscle.value, afferent.value)
        self.muscle = muscle
        self.afferent = afferent
        self.neuron_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=60
        )
        nest.Connect(
            pre=add_multimeter(self.name),
            post=self.neuron_ids
        )
