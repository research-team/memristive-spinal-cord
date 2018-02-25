import nest
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.toolkit.multimeter import add_multimeter


class AfferentFiber:

    def __init__(self, muscle: Muscle, afferent: Afferent):
        self.name = 'afferent_{}_fiber_{}'.format(muscle.value, afferent.value)
        self.muscle = muscle
        self.afferent = afferent
        self.neuron_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=60,
            params={
                't_ref': 2.,  # Refractory period
                'V_m': -70.0,  #
                'E_L': -70.0,  #
                'E_K': -77.0,  #
                'g_L': 30.0,  #
                'g_Na': 12000.0,  #
                'g_K': 3600.0,  #
                'C_m': 134.0,  # Capacity of membrane (pF)
                'tau_syn_ex': 0.5,  # Time of excitatory action (ms)
                'tau_syn_in': 5.0  # Time of inhibitory action (ms)
            }
        )
        nest.Connect(
            pre=add_multimeter(self.name),
            post=self.neuron_ids
        )
