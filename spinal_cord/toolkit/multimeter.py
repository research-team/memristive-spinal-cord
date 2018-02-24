import nest
from pkg_resources import resource_filename


def add_multimeter(name: str):
    return nest.Create(
        model='multimeter',
        n=1,
        params={
            'label': '{}/{}'.format(resource_filename('spinal_cord', 'results'), name),
            'record_from': ['V_m'],
            'withtime': True,
            'withgid': True,
            'interval': 0.1,
            'to_file': True,
            'to_memory': False
        })
