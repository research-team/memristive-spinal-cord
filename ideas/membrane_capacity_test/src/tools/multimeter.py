import nest
from pkg_resources import resource_filename


def add_multimeter(name: str) -> int:
    return nest.Create(
        model='multimeter',
        n=1,
        params={
            'label': '{}/{}/{}'.format(resource_filename('membrane_capacity_test', 'results'), 'raw_data', name),
            'record_from': ['V_m'],
            'withtime': True,
            'withgid': False,
            'interval': 0.1,
            'to_file': True,
            'to_memory': False
        })
