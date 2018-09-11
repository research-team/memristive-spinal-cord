from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 6
    RATE = 40
    SIMULATION_TIME = 175.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 8. # ms

    TO_PLOT = {
        'node1.1': 'Node 1.1',
        'node1.2': 'Node 1.2',
        'node1.3': 'Node 1.3',
        'hidden_1': 'Hidden Nuclei 1',
        'node2.1': 'Node 2.1',
        'node2.2': 'Node 2.2',
        'node2.3': 'Node 2.3',
        'node2.4': 'Node 2.4',
        'node2.5': 'Node 2.5',
        'node2.6': 'Node 2.6',
        'node3.1': 'Node 3.1',
        'node3.2': 'Node 3.2',
        'node3.3': 'Node 3.3',
        'node3.4': 'Node 3.4',
        'node3.5': 'Node 3.5',
        'node3.6': 'Node 3.6',
        'node4.1': 'Node 4.1',
        'node4.2': 'Node 4.2',
        'node4.3': 'Node 4.3',
        'node4.4': 'Node 4.4',
        'node4.5': 'Node 4.5',
        'node4.6': 'Node 4.6',
        'node5.1': 'Node 5.1',
        'node5.2': 'Node 5.2',
        'node5.3': 'Node 5.3',
        'node5.4': 'Node 5.4',
        'node5.5': 'Node 5.5',
        'node5.6': 'Node 5.6',
        'node6.1': 'Node 6.1',
        'node6.2': 'Node 6.2',
        'node6.3': 'Node 6.3',
        'node6.4': 'Node 6.4',
        'node6.5': 'Node 6.5',
        'pool1': 'Pool1',
        'pool2': 'Pool2',
        'pool3': 'Pool3',
        'pool4': 'Pool4',
        'pool5': 'Pool5',
        'pool6': 'Pool6',
        'moto': 'Moto',
        }

    TO_PLOT_WITH_SLICES = {
        'moto': 6
    }


def create(n: int):
    return Create(
        model='hh_cond_exp_traub',
        n=n,
        params={
            't_ref': 2.,
            'V_m': -70.0,
            'E_L': -70.0,
            'g_L': 75.0,
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
        pool = [create_with_mmeter(40, 'pool{}'.format(i)) for i in range(1, 7)]
        moto = create_with_mmeter(169, 'moto')
        ees = EES()
        ees.connect_ees(sensory)
        ees.connect_ees(moto)

        for pool_nucleus in pool:
            connect(pool_nucleus, moto, 25, 20)
        connect(ia_aff, moto, 25, 20)

        node11 = create_with_mmeter(40, 'node1.1')
        node12 = create_with_mmeter(40, 'node1.2')
        node13 = create_with_mmeter(40, 'node1.3')
        node14 = create_with_mmeter(40, 'node1.4')
        node15 = create_with_mmeter(40, 'node1.5')
        node16 = create_with_mmeter(40, 'node1.6')
        node17 = create_with_mmeter(40, 'node1.7')
        node18 = create_with_mmeter(40, 'node1.8')
        node19 = create_with_mmeter(40, 'node1.9')
        node0110 = create_with_mmeter(40, 'node1.010')
        node0111 = create_with_mmeter(40, 'node1.011')

        connect(node13, pool[0], 40., 30)
        connect(node14, pool[0], 40., 30)
        connect(node15, pool[0], 40., 30)
        connect(node16, pool[0], 40., 30)
        connect(node17, pool[0], 40., 30)
        connect(node18, pool[0], 40., 30)
        connect(node19, pool[0], 40., 30)
        connect(sensory, node11, 10., 40)
        connect(node11, node12, 15., 40, 2.)
        connect(node12, node13, 15., 40, 2.)
        connect(node13, node14, 15., 40, 2.)
        connect(node14, node15, 15., 40, 2.)
        connect(node15, node16, 15., 40, 2.)
        connect(node16, node17, 15., 40, 2.)
        connect(node17, node18, 15., 40, 2.)
        connect(node18, node19, 15., 40, 2.)
        connect(node19, node0110, 15., 40, 2.)
        connect(node0110, node0111, 15., 40, 2.)

        hidden_nuclei_1 = create_with_mmeter(40, 'hidden_1')
        node21 = create_with_mmeter(40, 'node2.1')
        node22 = create_with_mmeter(40, 'node2.2')
        node23 = create_with_mmeter(40, 'node2.3')
        node24 = create_with_mmeter(40, 'node2.4')
        node25 = create_with_mmeter(40, 'node2.5')
        node26 = create_with_mmeter(40, 'node2.6')

        connect(node11, node21, 15., 40, 2.)
        connect(node21, node22, 20., 40)
        connect(node22, node21, 20., 40)
        connect(node21, node23, 4., 40)
        connect(node23, hidden_nuclei_1, 15., 40, 2.)
        connect(hidden_nuclei_1, node24, 15., 40, 2.)
        connect(node24, node25, 15., 40)
        connect(node25, node26, 15., 40)

        connect(node11, node23, 7., 40, 0.1)
        connect(node24, pool[1], 80., 40)
        connect(node25, pool[1], 80., 40)
        connect(node26, pool[1], 80., 40)

        hidden_nuclei_2 = create_with_mmeter(40, 'hidden_2')
        node31 = create_with_mmeter(40, 'node3.1') # Why?
        node32 = create_with_mmeter(40, 'node3.2') # Why?
        node33 = create_with_mmeter(40, 'node3.3')
        node34 = create_with_mmeter(40, 'node3.4')
        node35 = create_with_mmeter(40, 'node3.5')
        node36 = create_with_mmeter(40, 'node3.6')

        connect(node23, node31, 15., 40)
        connect(node23, node33, 6., 40, .1)
        connect(node31, node32, 17., 40)
        connect(node32, node31, 17., 40)
        connect(node31, node33, 4., 40, 1.5)
        connect(node33, hidden_nuclei_2, 15., 40, 2.)
        connect(hidden_nuclei_2, node34, 15., 40)
        connect(node34, node35, 15., 40)
        connect(node35, node36, 15., 40)

        connect(node34, pool[2], 80., 40)
        connect(node35, pool[2], 80., 40)
        connect(node36, pool[2], 80., 40)

        connect(node33, node13, Params.INH_COEF.value * -40, 80, .1)

        hidden_nuclei_3 = create_with_mmeter(40, 'hidden_3')
        node41 = create_with_mmeter(40, 'node4.1')
        node42 = create_with_mmeter(40, 'node4.2')
        node43 = create_with_mmeter(40, 'node4.3')
        node44 = create_with_mmeter(40, 'node4.4')
        node45 = create_with_mmeter(40, 'node4.5')
        node46 = create_with_mmeter(40, 'node4.6')

        connect(node33, node41, 15., 40)
        connect(node33, node43, 6., 40, .1)
        connect(node41, node42, 17., 40)
        connect(node42, node41, 17., 40)
        connect(node41, node43, 4., 40)
        connect(node43, hidden_nuclei_3, 15., 40, 2.)
        connect(hidden_nuclei_3, node44, 15., 40, 2.)
        connect(node44, node45, 15., 40)
        connect(node45, node46, 15., 40)

        connect(node44, pool[3], 40., 60)
        connect(node45, pool[3], 60., 60)
        connect(node46, pool[3], 80., 60)

        connect(node43, node24, Params.INH_COEF.value * -30, 60, .1)

        node51 = create_with_mmeter(40, 'node5.1')
        node52 = create_with_mmeter(40, 'node5.2')
        node53 = create_with_mmeter(40, 'node5.3')
        node54 = create_with_mmeter(40, 'node5.4')
        node55 = create_with_mmeter(40, 'node5.5')
        node56 = create_with_mmeter(40, 'node5.6')

        connect(node43, node51, 15., 40)
        connect(node43, node53, 9., 40, .1)
        connect(node51, node52, 17., 40)
        connect(node52, node51, 17., 40)
        connect(node51, node53, 4., 40)
        connect(node53, node54, 15., 40, 2.)
        connect(node54, node55, 15., 40)
        connect(node55, node56, 15., 40)

        connect(node54, pool[4], 40., 40)
        connect(node55, pool[4], 60., 40)
        connect(node56, pool[4], 80., 40)

        connect(node53, node34, Params.INH_COEF.value * -30, 60, .1)

        node61 = create_with_mmeter(40, 'node6.1')
        node62 = create_with_mmeter(40, 'node6.2')
        node63 = create_with_mmeter(40, 'node6.3')
        node64 = create_with_mmeter(40, 'node6.4')
        node65 = create_with_mmeter(40, 'node6.5')

        connect(node53, node61, 15., 40)
        connect(node53, node63, 6., 40, .1)
        connect(node61, node62, 17., 40)
        connect(node62, node61, 17., 40)
        connect(node61, node63, 4., 40)
        connect(node63, node64, 15., 40, 2.)
        connect(node64, node65, 15., 40, 2.)

        connect(node64, pool[5], 60., 40)
        connect(node65, pool[5], 80., 40)

        connect(node63, node45, Params.INH_COEF.value * -25, 60, .1)
        connect(node63, node54, Params.INH_COEF.value * -25, 60, .1)