import shutil
from memristive_spinal_cord.layer2.schemes.basic.components.parameters import Paths


class ToolKit():

    def clear_results(self):
        path = Paths.RESULTS_DIR.value

        try:
            shutil.rmtree(path=path)
        except FileNotFoundError:
            pass

    def plot_left_column(self):
        pass # TODO implement plot_left_column()

    def plot_interneuronal_pool(self):
        pass # TODO implement plot_interneuronal_pool()