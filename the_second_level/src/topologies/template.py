from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 7
    RATE = 40
    SIMULATION_TIME = 50.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 12. # ms

    TO_PLOT = {'neurons': 'Neurons'}

    TO_PLOT_WITH_SLICES = {}


def create(n: int):
    return Create(
        model='hh_cond_exp_traub',
        n=n,
        params={
            't_ref': 2.,
            'V_m': -70.0,
            'E_L': -70.0,
            'g_L': 50.0,
            'tau_syn_ex': .2,
            'tau_syn_in': 1.})

def create_with_mmeter(n: int, name: str):
    gids = create(n)
    Connect(pre=add_multimeter(name), post=gids)
    return gids


def connect(pre, post, weight, degree, delay=1.):
    Connect(
        pre=pre,
        post=post,
        syn_spec={
            'model': 'static_synapse',
            'delay': delay,
            'weight': weight,
        },
        conn_spec={
            'rule': 'fixed_outdegree',
            'outdegree': degree,
            'multapses': True,
            'autapses': True
        })

class Topology():
    def __init__(self):
        self.sublevels = [Sublevel(i) for i in range(Params.NUM_SUBLEVELS.value)]
        ees = Create(
            model='spike_generator',
            params={
                'spike_times': [10. + i * period for i in range(Params.NUM_SPIKES.value)],
                'spike_weights': [300. for i in range(Params.NUM_SPIKES.value)]})

        neurons = create_with_mmeter(100, 'Neuron')
        poisson_gen = Create(
            model='poisson_generator', params={'rate': 100.})
        Connect(
            pre=poisson_gen,
            post=neurons,
            syn_spec={'weight': 300.})