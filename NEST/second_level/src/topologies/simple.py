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

    TO_PLOT = {
        'left0': 'Left 0',
        'right0': 'Right 0',
        'left1': 'Left 1',
        'right1': 'Right 1',
        'left2': 'Left 2',
        'right2': 'Right 2',
        'left3': 'Left 3',
        'right3': 'Right 3',
        'left4': 'Left 4',
        'right4': 'Right 4',
        'left5': 'Left 5',
        'right5': 'Right 5',
        'interneuronal_pool': 'Interneuronal Pool',
        'motor_pool': 'Motor Pool'}

    TO_PLOT_WITH_SLICES = {
        'motor_pool': 7
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


class EES:
    def __init__(self):
        self.ees = Create(
            model='spike_generator',
            params={
                'spike_times': [10. + i * round(1000. / Params.RATE.value, 1) for i in range(Params.NUM_SPIKES.value)],
                'spike_weights': [500. for i in range(Params.NUM_SPIKES.value)]})

    def connect(cls, post):
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

class Sublevel:
    def __init__(self, index: int):
        self.right = create_with_mmeter(40, 'right{}'.format(index))
        self.left = create_with_mmeter(40, 'left{}'.format(index))
        connect(self.right, self.left, 25., 25)
        connect(self.left, self.right, 25., 25)

class Level2:
    def __init__(self):
        self.sublevels = [Sublevel(i) for i in range(Params.NUM_SUBLEVELS.value)]
        for i in range(Params.NUM_SUBLEVELS.value-1):
            connect(self.sublevels[i].right, self.sublevels[i+1].right, 13., 15)
            connect(self.sublevels[i+1].left, self.sublevels[i].left, 10., 15)
            connect(self.sublevels[i+1].right, self.sublevels[i].left, -20., 20)
    def connect_all_to(self, post):
        for sublevel in self.sublevels:
            connect(sublevel.left, post, 35., 35)

class Topology:
    def __init__(self):
        ees = EES()
        pool = create_with_mmeter(100, 'interneuronal_pool')
        moto = create_with_mmeter(169, 'motor_pool')
        level2 = Level2()
        level2.connect_all_to(pool)
        connect(pool, moto, 25., 25)
        ees.connect(level2.sublevels[0].right)
        ees.connect(moto)
