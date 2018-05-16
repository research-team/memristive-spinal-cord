from nest import Create, Connect
from rybak_affs.src.afferents.afferent_fiber import AfferentFiber
from rybak_affs.src.namespace import Muscle, Afferent
from rybak_affs.src.tools.multimeter import add_multimeter


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
                'tau_syn_in': .5})

def connect(pre, post, weight, degree):
    Connect(
        pre=pre,
        post=post,
        syn_spec={
            'model': 'static_synapse',
            'delay': 1.,
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

        connect(pre=self.general_right, post=self.general_left, weight=50., degree=10)
        connect(pre=self.hidden_right, post=self.hidden_left, weight=50., degree=10)
        connect(pre=self.hidden_left, post=self.hidden_right, weight=50., degree=10)

        connect(pre=self.general_right, post=self.hidden_right, weight=50., degree=10)
        connect(pre=self.hidden_left, post=self.inh, weight=50., degree=10)
        connect(pre=self.inh, post=self.general_left, weight=-50., degree=10)

class Level2:
    def __init__(self):
        self.sublayers = [Sublayer(i) for i in range(6)]
        for i in range(5):
            connect(
                pre=self.sublayers[i].general_right,
                post=self.sublayers[i+1].general_right,
                weight=10., degree=10)
            connect(
                pre=self.sublayers[i].hidden_right,
                post=self.sublayers[i+1].general_right,
                weight=15., degree=10)
            connect(
                pre=self.sublayers[i].general_left,
                post=self.sublayers[i+1].general_left,
                weight=50., degree=10)
 

ia_aff = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.IA)
moto = create(n=169)
Connect(pre=add_multimeter('moto'), post=moto)
ia_int = create(n=196)
Connect(pre=add_multimeter('ia_int'), post=ia_int)
rc = create(196)
Connect(add_multimeter('rc'), post=rc)
level2 = Level2()

connect(ia_aff.neuron_ids, level2.sublayers[0].general_right, 30., 20)
connect(ia_aff.neuron_ids, moto, 5., 196)
connect(ia_aff.neuron_ids, ia_int, 12., 20)
connect(moto, rc, 7., 100)
connect(rc, moto, -7., 100)
connect(ia_int, moto, -7., 100)
