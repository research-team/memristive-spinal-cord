from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 1
    RATE = 40
    SIMULATION_TIME = 150.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 8. # ms

    TO_PLOT = {
        'node1.1': 'Node 1.1',
        'node1.2': 'Node 1.2',
        'node1.3': 'Node 1.3',
        'pool': 'Pool',
        'moto': 'Moto',
        }

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
            'tau_syn_ex': .2,
            'tau_syn_in': 3.})

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

class Node:
    def __init__(self, index):
        self.node1 = create_with_mmeter(40, 'node{}.1'.format(index))
        self.node2 = create_with_mmeter(40, 'node{}.2'.format(index))
        self.node3 = create_with_mmeter(40, 'node{}.3'.format(index))
        self.node4 = create_with_mmeter(40, 'node{}.4'.format(index))
        self.node5 = create_with_mmeter(40, 'node{}.5'.format(index))
        self.node6 = create_with_mmeter(40, 'node{}.6'.format(index))

        connect(self.node1, self.node2, 15., 30)
        connect(self.node2, self.node1, 15., 30)
        connect(self.node1, self.node3, 10., 30)
        connect(self.node2, self.node3, -10., 40)
        connect(self.node3, self.node4, 10., 30)
        connect(self.node4, self.node5, 10., 30)
        connect(self.node5, self.node6, 10., 30)

class Topology:
    def __init__(self):

        sensory = create_with_mmeter(60, 'sensory')
        ia_aff = create_with_mmeter(169, 'ia_aff')
        pool = create_with_mmeter(100, 'pool')
        moto = create_with_mmeter(169, 'moto')
        ees = EES()
        ees.connect_ees(sensory)
        ees.connect_ees(moto)

        connect(pool, moto, 5, 100)
        connect(ia_aff, moto, 5, 100)

        node11 = create_with_mmeter(40, 'node1.1')
        node12 = create_with_mmeter(40, 'node1.2')
        node13 = create_with_mmeter(40, 'node1.3')

        connect(sensory, node11, 10., 40)
        connect(node11, node12, 15., 40)
        connect(node12, node13, 15., 40)
        connect(node13, pool, 30., 40)
