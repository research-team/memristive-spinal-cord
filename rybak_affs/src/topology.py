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

ia_aff = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.IA)
moto = create(n=169)
Connect(pre=add_multimeter('moto'), post=moto)
ia_int = create(n=196)
Connect(pre=add_multimeter('ia_int'), post=ia_int)
rc = create(196)
Connect(add_multimeter('rc'), post=rc)

connect(ia_aff.neuron_ids, moto, 5., 196)
connect(ia_aff.neuron_ids, ia_int, 12., 100)
connect(moto, rc, 7., 100)
connect(rc, moto, -7., 100)
connect(ia_int, moto, -7., 100)