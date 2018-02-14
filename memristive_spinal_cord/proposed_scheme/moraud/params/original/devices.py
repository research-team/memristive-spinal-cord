import definitions
from memristive_spinal_cord.proposed_scheme.moraud.entities import Layer1Multimeters

device_params = dict()
storage_dir = definitions.RESULTS_DIR


def add_multimeter_params(multimeter):
    device_params[multimeter] = dict(
        model='multimeter',
        n=1,
        params={
            'label': storage_dir + '/' + multimeter.value,
            'record_from': ['V_m'],
            'withtime': True,
            'withgid': True,
            'interval': 0.2,
            'to_file': True,
            'to_memory': False,
        }
    )


for multimeter in Layer1Multimeters:
    add_multimeter_params(multimeter)
