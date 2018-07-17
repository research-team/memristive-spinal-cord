from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter
from the_second_level.src.tools.spike_detector import add_spike_detector


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 2
    RATE = 40
    SIMULATION_TIME = 100.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 12. # ms

    TO_PLOT = [
        'sensory',
        'node1.1',
        'node1.2',
        'node1.3',
        'node1.4',
        'node2.1',
        'node2.2',
        'node2.3',
        'node2.4',
        'node2.5',
        'node2.6',
        'node2.7',
        'pool',
        'moto'
    ]

    TO_PLOT_WITH_SLICES = {
        'moto': 2
    }


def create(n: int):
    return Create(
        model='hh_cond_exp_traub',
        n=n,
        params={
            't_ref': 2.,
            'V_m': -70.0,
            'E_L': -70.0,
            'g_L': 100.0,
            'tau_syn_ex': .5,
            'tau_syn_in': 1.})

def create_with_mmeter(n: int, name: str):
    gids = create(n)
    Connect(pre=add_multimeter(name), post=gids)
    Connect(pre=gids, post=add_spike_detector(name))
    return gids


def connect(pre, post, weight, delay=1.):
    Connect(
        pre=pre,
        post=post,
        syn_spec={
            'model': 'static_synapse',
            'delay': delay,
            'weight': weight,
        },
        conn_spec={
            'rule': 'one_to_one',
        })


class EES:
    def __init__(self):
        self.ees = Create(
            model='spike_generator',
            params={
                'spike_times': [10. + i * round(1000. / Params.RATE.value, 1) for i in range(Params.NUM_SPIKES.value)],
                'spike_weights': [500. for i in range(Params.NUM_SPIKES.value)]})

    def connect_ees(cls, post):
        Connect(
            pre=cls.ees,
            post=post,
            syn_spec={
                'model': 'static_synapse',
                'weight': 1.,
                'delay': .1
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': len(post),
                'autapses': False,
                'multapses': False
            })

class Topology:
    def __init__(self):

        sensory = create_with_mmeter(1, 'sensory')
        ia_aff = create_with_mmeter(1, 'ia_aff')
        moto = create_with_mmeter(1, 'moto')
        pool = create_with_mmeter(1, 'pool')

        ees = EES()
        ees.connect_ees(sensory)

        connect(pool, moto, 100.)

        # the first sublevel
        node11 = create_with_mmeter(1, 'node1.1')
        node12 = create_with_mmeter(1, 'node1.2')
        node13 = create_with_mmeter(1, 'node1.3')
        node14 = create_with_mmeter(1, 'node1.4')

        connect(sensory, node11, 200.)
        connect(node11, node12, 200., 2.)
        connect(node12, node13, 200., 2.)
        connect(node13, node14, 200.)
        connect(node14, pool, 200.)

        #the second sublevel
        node21 = create_with_mmeter(1, 'node2.1')
        node22 = create_with_mmeter(1, 'node2.2')
        node23 = create_with_mmeter(1, 'node2.3')
        node24 = create_with_mmeter(1, 'node2.4')
        node25 = create_with_mmeter(1, 'node2.5')
        node26 = create_with_mmeter(1, 'node2.6')
        node27 = create_with_mmeter(1, 'node2.7')

        connect(node11, node21, 200.)
        connect(node11, node23, 120., .1)
        connect(node21, node22, 205.)
        connect(node22, node21, 205.)
        connect(node22, node23, 100.)
        connect(node23, node24, 200.)
        connect(node24, node25, 200.)
        connect(node25, node26, 200.)
        connect(node26, node27, 200.)
