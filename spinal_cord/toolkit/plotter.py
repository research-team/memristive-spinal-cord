import shutil
from pkg_resources import resource_filename
import os


def clear_results():
    results_dir_filename = resource_filename('spinal_cord', 'results')
    if os.path.isdir(results_dir_filename):
        shutil.rmtree(results_dir_filename)
        os.mkdir(results_dir_filename)
    else:
        os.mkdir(results_dir_filename)

