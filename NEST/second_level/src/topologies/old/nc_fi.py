from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 2
    RATE = 40
    SIMULATION_TIME = 100.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 10. # ms
    NUM_NEURONS = 40
    DEGREE = 3

    TO_PLOT_SUB_0 = ['node0.{}'.format(i) for i in range(6)]
    TO_PLOT_SUB_1 = ['node1.{}'.format(i) for i in range(7)]
    TO_PLOT_SUB_2 = ['node2.{}'.format(i) for i in range(7)]
    TO_PLOT_POOL = ['pool{}'.format(i) for i in range(6)]
    TO_PLOT_MOTO = ['moto{}'.format(i) for i in range(6)]
    TO_PLOT = ['sensory', 'ia_aff']
    TO_PLOT.extend(TO_PLOT_SUB_0)
    TO_PLOT.extend(TO_PLOT_SUB_1)
    TO_PLOT.extend(TO_PLOT_SUB_2)
    TO_PLOT.extend(TO_PLOT_POOL)
    TO_PLOT.extend(TO_PLOT_MOTO)

    TO_PLOT_WITH_SLICES = {'moto': NUM_SPIKES}


def create(n: int):
    return Create(
        model='hh_cond_exp_traub',
        n=n,
        params={
            't_ref': 1.,
            'V_m': -70.0,
            'E_L': -70.0,
            'g_L': 100.0,
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
            'rule': 'fixed_indegree',
            'indegree': degree,
            'multapses': True,
            'autapses': True
        })

class Sublevel:
    def __init__(self, index: int):
        self.index = index
        self.nodes = []

    def add_nodes(self, n: int):
        for i in range(n):
            self.add_node(i)

    def add_node(self, subindex: int):
        self.nodes.append(create_with_mmeter(Params.NUM_NEURONS.value, 'node{}.{}'.format(self.index, subindex)))

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
        fi = Params.DEGREE.value # fixed indegree
        we = 175 # weight

        sensory = create_with_mmeter(60, 'sensory')
        ia_aff = create_with_mmeter(196, 'ia_aff')
        moto = [create_with_mmeter(25, 'moto{}'.format(i)) for i in range(6)]
        pool = [create_with_mmeter(Params.NUM_NEURONS.value, 'pool{}'.format(i)) for i in range(6)]

        for i in range(6):
            connect(pool[i], moto[i], we, fi)

        for m in moto:
            connect(ia_aff, m, we, fi)

        ees = EES()
        ees.connect_ees(sensory)
        ees.connect_ees(ia_aff)

        sublevels = [Sublevel(i) for i in range(6)]
        
        # The first sublevel 

        sublevels[0].add_nodes(6)
        n = sublevels[0].nodes
        connect(sensory, n[0], we, fi)

        connect(n[0], n[1], we, fi)
        connect(n[1], n[2], we, fi)
        connect(n[2], n[3], we, fi, 4.)
        connect(n[3], n[4], we, fi, 2.)
        connect(n[4], n[5], we, fi, 2.)

        connect(n[2], pool[0], we, fi, 2.)
        connect(n[2], pool[1], we, fi, 2.)

        connect(n[3], pool[2], we, fi, 2.)
        connect(n[4], pool[3], we, fi, 2.)
        connect(n[5], pool[4], we, fi, 2.)

        # The second sublevel

        sublevels[1].add_nodes(7)
        n = sublevels[1].nodes

        connect(n[0], n[1], we, fi)
        connect(n[1], n[0], we, fi)
        connect(n[0], n[2], we * 0.5, fi)
        connect(sublevels[0].nodes[0], n[2], we * 0.45, fi, 0.1)
        connect(sublevels[0].nodes[0], n[0], we, fi)

        connect(n[2], n[3], we, fi)
        connect(n[3], n[4], we, fi)
        connect(n[4], n[5], we, fi)
        connect(n[5], n[6], we, fi)

        connect(n[4], pool[1], we, fi)
        connect(n[5], pool[2], we, fi)
        connect(n[6], pool[3], we, fi)
        connect(n[4], pool[4], we, fi)
        connect(n[3], pool[5], we, fi)

        # The third sublevel

        sublevels[2].add_nodes(7)
        n = sublevels[2].nodes

        connect(n[0], n[1], we, fi)
        connect(n[1], n[0], we, fi)
        connect(n[0], n[2], we * 0.45, fi)
        connect(sublevels[1].nodes[1], n[2], we * 0.45, fi, 0.1)
        connect(sublevels[1].nodes[1], n[0], we, fi)

        connect(n[2], n[3], we, fi)
        connect(n[3], n[4], we, fi)
        connect(n[4], n[5], we, fi)
        connect(n[5], n[6], we, fi)

        connect(n[4], pool[1], we, fi)
        connect(n[5], pool[2], we, fi)
        connect(n[6], pool[3], we, fi)
        connect(n[4], pool[4], we, fi)
        connect(n[3], pool[5], we, fi)
        connect(n[0], sublevels[0].nodes[2], -we, fi)