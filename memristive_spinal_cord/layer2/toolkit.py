import shutil


class ToolKit():

    @staticmethod
    def clear_results(path):
        try:
            shutil.rmtree(path=path)
        except FileNotFoundError:
            pass

    def plot_left_column(self):
        pass # TODO implement plot_left_column()

    def plot_interneuronal_pool(self):
        pass # TODO implement plot_interneuronal_pool()

    def plot_hidden_layers(self):
        pass # TODO implement plot_hidden_layers()