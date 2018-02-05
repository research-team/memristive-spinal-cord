import pylab
import shutil
import os
from multiprocessing import Pool


class Plotter:

    @staticmethod
    def get_filenames() -> list:
        print(__file__)
        print(os.path.dirname(os.path.abspath(__file__)))
        return [filename for filename in os.listdir(os.path.dirname(os.path.abspath(__file__))) if
                filename.split('.')[-1] == 'txt']

    def __init__(self):
        self.path_to_images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'images'))
        if os.path.isdir(self.path_to_images_dir):
            shutil.rmtree(self.path_to_images_dir)
            os.mkdir(self.path_to_images_dir)
        else:
            os.mkdir(self.path_to_images_dir)

    def plot(self, filename: str) -> None:
        with open(filename) as file:
            os.mkdir(os.path.join(self.path_to_images_dir, filename.split('.')[0]))
            aff_count = 1
            for line in file.readlines():
                if aff_count <= 60:
                    freqs = [float(value) for value in line.split()]
                    times = [20 * i for i in range(len(freqs))]
                    print('Plotting {}: afferent {}'.format(filename, aff_count))
                    pylab.plot(times, freqs)
                    pylab.xlabel('Time (ms)')
                    pylab.ylabel('Frequency (Hz)')
                    pylab.savefig('{}/{}aff{}.png'.format(
                        os.path.join(self.path_to_images_dir, filename.split('.')[0]),
                        filename.split('.')[0], aff_count))
                    pylab.close('all')
                    aff_count += 1

    def plot_all(self):
        pool = Pool()
        pool.map(self.plot, self.get_filenames())
        pool.close()
        pool.join()


if __name__ == '__main__':
    plotter = Plotter()
    plotter.plot_all()
