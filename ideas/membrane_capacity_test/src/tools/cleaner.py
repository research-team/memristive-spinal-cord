import os
import shutil
from pkg_resources import resource_filename


def clean():
    results_dir = resource_filename('membrane_capacity_test', 'results')
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    print('Hey, {}!'.format(os.path.join(results_dir, 'raw_data')))
    os.makedirs(os.path.join(results_dir, 'raw_data'))
    os.mkdir(os.path.join(results_dir, 'img'))
