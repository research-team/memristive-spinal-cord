import nest
from rybak_affs.src.namespace import Muscle, Afferent
from rybak_affs.src.tools.multimeter import add_multimeter
from rybak_affs.src.afferents.receptor import Receptor, DummySensoryReceptor
from random import randint


class AfferentFiber:

    def __init__(self, muscle: Muscle, afferent: Afferent):
        self.name = 'afferent_{}_fiber_{}'.format(muscle.value, afferent.value)
        self.muscle = muscle
        self.afferent = afferent
        self.neuron_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=60,
            params={
                't_ref': 2.,
                'V_m': -70.0,
                'E_L': -70.0,
                'g_L': 100.0,
                'tau_syn_ex': .2,
                'tau_syn_in': .5
            }
        )
        self.receptor = Receptor(
            muscle=muscle, afferent=afferent
        )
        nest.Connect(
            pre=self.receptor.receptor_ids,
            post=self.neuron_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 500.
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=add_multimeter(self.name),
            post=self.neuron_ids
        )


class DummySensoryAfferentFiber:
    def __init__(self, dummy_sensory_receptor: DummySensoryReceptor):
        self.name = 'dummy_sensory'
        self.neuron_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=60,
            params={
                't_ref': 2.,
                'V_m': -70.0,
                'E_L': -70.0,
                'g_L': 100.0,
                'tau_syn_ex': .2,
                'tau_syn_in': .5
            }
        )
        self.receptor = dummy_sensory_receptor
        nest.Connect(
            pre=self.receptor.receptor_id,
            post=self.neuron_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 0.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 20,
                'multapses': False
            }
        )
        nest.Connect(
            pre=add_multimeter(self.name),
            post=self.neuron_ids
        )
