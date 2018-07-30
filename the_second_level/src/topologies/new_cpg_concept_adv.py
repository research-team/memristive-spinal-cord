from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 7
    RATE = 40
    SIMULATION_TIME = 200.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 9. # ms

    plot_pool = ['pool{}'.format(i) for i in range(1, 7)]
    plot_sublevel1 = ['node1.{}'.format(i) for i in range(7)]
    TO_PLOT = ['sensory', 'ia_aff', 'moto']
    TO_PLOT.extend(plot_pool)
    TO_PLOT.extend(plot_sublevel1)

    TO_PLOT_WITH_SLICES = {
        'moto': 7
    }


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

def connect_ia_aff(ia_aff, moto):
    Connect(
            pre=ia_aff,
            post=moto,
            syn_spec={
                'model': 'static_synapse',
                'weight': 50,
                'delay': .1
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 169,
                'autapses': False,
                'multapses': False
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

        sensory = create_with_mmeter(60, 'sensory')
        ia_aff = create_with_mmeter(169, 'ia_aff')
        pool = [create_with_mmeter(40, 'pool{}'.format(i)) for i in range(1, 7)]
        moto = create_with_mmeter(169, 'moto')
        ees = EES()
        ees.connect_ees(sensory)
        ees.connect_ees(ia_aff)
        connect_ia_aff(ia_aff, moto)

        sublevel1 = [create_with_mmeter(40, 'node1.{}'.format(i)) for i in range(9)]
        for i in range(len(sublevel1)-1):
            connect(sublevel1[i], sublevel1[i+1], 150., 4)
        for i in range(2, len(sublevel1)):
            connect(sublevel1[i], pool[0], 50., 3)
        for i in range(len(pool)):
            connect(pool[0], moto, 100, 3)

        connect(sensory, sublevel1[0], 100., 5)