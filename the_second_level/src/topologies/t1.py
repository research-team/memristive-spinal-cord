from enum import Enum
from nest import Create, Connect
from the_second_level.src.tools.multimeter import add_multimeter


class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 7
    RATE = 40
    SIMULATION_TIME = 150.
    INH_COEF = 1.
    PLOT_SLICES_SHIFT = 12. # ms

    TO_PLOT = {'sensory': 'Sensory'}

    TO_PLOT_WITH_SLICES = {}

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

class Sublevel():
    def __init__(self, index: int):
        self.right = create_with_mmeter(100, 'right{}'.format(index))
        self.left = create_with_mmeter(100, 'left{}'.format(index))
        connect(pre=self.right, post=self.left, weight=10., degree=15)
        connect(pre=self.left, post=self.right, weight=10., degree=15)

class Level2():
    def __init__(self):
        self.sublevels = [Sublevel(i) for i in range(Params.NUM_SUBLEVELS.value)]
        for i in range(len(self.sublevels)-1):
            connect(pre=self.sublevels[i].right, post=self.sublevels[i+1].right, weight=10, degree=15)
            connect(pre=self.sublevels[i+1].left, post=self.sublevels[i].left, weight=10, degree=15)


class Topology():
    def __init__(self):
        period = round(1000. / Params.RATE.value, 1)
        ees = Create(
            model='spike_generator',
            params={
                'spike_times': [10. + i * period for i in range(Params.NUM_SPIKES.value)],
                'spike_weights': [500. for i in range(Params.NUM_SPIKES.value)]})

        level2 = Level2()
        sensory = create_with_mmeter(60, 'sensory')
        inter_pool = create_with_mmeter(200, 'Interneuronal Pool')
        motor_pool = create_with_mmeter(169, 'Motor Pool')

        for sublevel in level2.sublevels:
            connect(pre=sublevel.left, post=inter_pool, weight=10., degree=15)
        connect(pre=sensory, post=level2.sublevels[0].right, weight=10., degree=15)
        Connect(
            pre=ees,
            post=sensory,
            syn_spec={
                'model': 'static_synapse',
                'weight': 1.,
                'delay': .1
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 60,
                'autapses': False,
                'multapses': False
            })
