import os
import shutil
from ows.src.paths import results_path

class Cleaner:
    dirs_to_create = ['img', 'raw_data']

    @classmethod
    def clean(cls):
        if os.path.isdir(results_path):
            shutil.rmtree(results_path)

    @classmethod
    def create_structure(cls):
        if not os.path.isdir(results_path):
            for dir_name in cls.dirs_to_create:
                os.makedirs(os.path.join(results_path, dir_name))
