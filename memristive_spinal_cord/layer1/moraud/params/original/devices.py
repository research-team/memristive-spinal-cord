import definitions
from memristive_spinal_cord.layer1.moraud.entities import Layer1Multimeters

device_params = dict()
storage_dir = definitions.RESULTS_DIR,


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


add_multimeter_params(Layer1Multimeters.FLEX_INTER_1A)
