from enum import Enum
from nest import Create, Connect
from the_second_level.src.afferents.afferent_fiber import AfferentFiber
from the_second_level.src.namespace import Muscle, Afferent
from the_second_level.src.tools.multimeter import add_multimeter
from the_second_level.src.params import num_sublevels, inh_coef, rate, num_spikes

class Params(Enum):
    NUM_SUBLEVELS = 6
    NUM_SPIKES = 7
    RATE = 40
    SIMULATION_TIME = round(1000 / rate * 8, 1)
    INH_COEF = .4
    PLOT_SLICES_SHIFT = 12. # ms


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
















class Sublayer:
    def __init__(self, index: int):
        self.general_right = create(40)
        self.general_left = create(40)
        self.hidden_right = create(40)
        self.hidden_left = create(40)
        self.inh = create(40)

        connect(pre=self.general_right, post=self.general_left, weight=12., degree=40)
        connect(pre=self.hidden_right, post=self.hidden_left, weight=17., degree=40)
        connect(pre=self.hidden_left, post=self.hidden_right, weight=17., degree=40)

        connect(pre=self.general_right, post=self.hidden_right, weight=10., degree=40)
        connect(pre=self.hidden_left, post=self.inh, weight=16., degree=40)
        connect(pre=self.inh, post=self.general_left, weight=-13. * inh_coef, degree=80)

        Connect(pre=add_multimeter('general_right{}'.format(index)), post=self.general_right)
        Connect(pre=add_multimeter('general_left{}'.format(index)), post=self.general_left)
        Connect(pre=add_multimeter('hidden_right{}'.format(index)), post=self.hidden_right)
        Connect(pre=add_multimeter('hidden_left{}'.format(index)), post=self.hidden_left)
        Connect(pre=add_multimeter('inh{}'.format(index)), post=self.inh)

class Level2:
    def __init__(self):
        self.sublayers = [Sublayer(i) for i in range(num_sublevels)]
        for i in range(num_sublevels-1):
            connect(
                pre=self.sublayers[i].general_right,
                post=self.sublayers[i+1].general_right,
                weight=5., degree=40)
            connect(
                pre=self.sublayers[i].hidden_right,
                post=self.sublayers[i+1].general_right,
                weight=3., degree=40)
            connect(
                pre=self.sublayers[i].general_left,
                post=self.sublayers[i+1].general_left,
                weight=16., degree=40)

# ia_aff = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.IA)
try:
    type(ia_aff)
except NameError:
    ia_aff = create(60)
    Connect(pre=add_multimeter('afferent'), post=ia_aff)

period = round(1000. / rate, 1)

ees = Create(
    model='spike_generator',
    params={
        'spike_times': [10. + i * period for i in range(num_spikes)],
        'spike_weights': [300. for i in range(num_spikes)]
    })

Connect(
    pre=ees,
    post=ia_aff,
    syn_spec={
        'model': 'static_synapse',
        'weight': 1.,
        'delay': .1},
    conn_spec={
        'rule': 'fixed_outdegree',
        'outdegree': 60,
        'multapses': False,
        'autapses': False
    })

moto = create(n=169)
Connect(pre=add_multimeter('moto'), post=moto)
ia_int = create(n=196)
Connect(pre=add_multimeter('ia_int'), post=ia_int)
rc = create(196)
Connect(add_multimeter('rc'), post=rc)
level2 = Level2()

for i in range(num_sublevels):
    connect(pre=level2.sublayers[i].general_left, post=moto, weight=15., degree=100)

# connect(ia_aff.neuron_ids, level2.sublayers[0].general_right, 20., 20, 3.)
# connect(ia_aff.neuron_ids, moto, 7., 196)
# connect(ia_aff.neuron_ids, ia_int, 3., 196)

connect(ia_aff, level2.sublayers[0].general_right, 30., 20, 3.)
connect(ia_aff, moto, 7., 196)
connect(ia_aff, ia_int, 3., 196)
connect(moto, rc, 7., 100)
connect(rc, moto, -7., 100)
connect(ia_int, moto, -7., 100)
