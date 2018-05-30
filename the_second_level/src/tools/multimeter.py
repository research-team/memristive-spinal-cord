import nest
import os
from rybak_affs.src.paths import raw_data_path


def add_multimeter(name):
    return nest.Create(
        model='multimeter',
        n=1,
        params={
            'label': os.path.join(raw_data_path, name),
            'record_from': ['V_m'],
            'withgid': True,
            'withtime': True,
            'interval': 0.1,
            'to_file': True,
            'to_memory': False})
