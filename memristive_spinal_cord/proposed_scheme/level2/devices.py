import definitions


device_params = dict()
storage_dir = definitions.RESULTS_DIR


def add_multimeter_params(multimeter: str):
    device_params[multimeter] = dict(
        model='multimeter',
        n=1,
        params={
            'label': storage_dir + '/' + multimeter,
            'record_from': ['V_m'],
            'withtime': True,
            'withgid': True,
            'interval': 0.2,
            'to_file': True,
            'to_memory': False,
        }
    )


for tier in range(1, 7):
    for exc in range(5):
        add_multimeter_params('Tier{}E{}-multimeter'.format(tier, exc))
    add_multimeter_params('Tier{}I0-multimeter'.format(tier))

for exc in range(2):
    add_multimeter_params('Tier0E{}-multimeter'.format(exc))
add_multimeter_params('Tier0I0-multimeter')

for pool in range(2):
    add_multimeter_params('Pool{}-multimeter'.format(pool))