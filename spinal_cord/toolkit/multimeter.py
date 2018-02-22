import nest
from pkg_resources import resource_filename


def add_multimeter(name: str):
    nest.Create(
        model='multimeter',
        n=1,
        params={
            'label': '{}/results/{}'.format(resource_filename('spinal_cord'), name),
            'record_from': ['V_m'],
            'withtime': True,
            'withgid': True,
            'interval': 0.1,
            'to_file': True,
            'to_memory': False
        })
