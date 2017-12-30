import shutil
import os


class ToolKit():

    def __init__(self, path, dirname):
        self.path = path
        self.dirname = dirname

    def clear_results(self):
        try:
            shutil.rmtree(path=os.path.join(self.path, self.dirname))
        except FileNotFoundError:
            pass

    def plot_left_column(self):
        pass # TODO implement plot_left_column()

    def plot_interneuronal_pool(self):
        pass # TODO implement plot_interneuronal_pool()