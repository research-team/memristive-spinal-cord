from memristive_spinal_cord.layer2.toolkit import ToolKit
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Constants
import os
import pylab


class HiddenTiersToolKit(ToolKit):
    def plot_hidden_layers(self, show_results: bool=False):
        figures_dirname = 'hidden_tiers'
        path = os.path.join(self.path, self.figures_dirname, figures_dirname)
        os.mkdir(path=path)
        for hidden_tier in range(1, 6):
            count = 0
            for type in ['Excitatory', 'Inhibitory']:
                if type == 'Excitatory':
                    neurotransmitter = Neurotransmitters.GLU.value
                else:
                    neurotransmitter = Neurotransmitters.GABA.value
                for side in ['Left', 'Right']:
                    raw_data_file = os.path.join(self.raw_data_dirname, 'HiddenTier{}{}{} [{}].dat'.format(
                        str(hidden_tier), side, type, neurotransmitter
                    ))
                    with open(raw_data_file) as raw_data:
                        voltage = []
                        time = []
                        for line in raw_data.readlines():
                            time.append(float(line.split()[1]))
                            voltage.append(float(line.split()[2]))
                    count += 1
                    pylab.subplot(4, 1, count)
                    pylab.axis([0, int(Constants.SIMULATION_TIME.value), -70, 50])
                    pylab.plot(time, voltage)
                    pylab.title('HiddenTier{}{}{}'.format(str(hidden_tier), side, type))
            if show_results:
                pylab.show()
            else:
                pylab.savefig(fname=os.path.join(path, 'HiddenTier{}'.format(str(hidden_tier))))
            pylab.close('all')
