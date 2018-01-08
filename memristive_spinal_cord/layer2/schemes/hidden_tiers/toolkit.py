from memristive_spinal_cord.layer2.toolkit import ToolKit
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Constants
import os
import pylab


class HiddenTiersToolKit(ToolKit):
    def plot_hidden_layers(self, *tiers, show_results: bool=False):
        for hidden_tier in tiers:
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
                    pylab.subplots_adjust(
                        left=0.07,
                        right=0.99,
                        bottom=0.03,
                        top=0.97,
                        hspace=0.30
                    )
            if show_results:
                pylab.show()
            else:
                figures_dirname = 'hidden_tiers'
                path = os.path.join(self.path, self.figures_dirname)
                if not os.path.isdir(path):
                    os.mkdir(path=path)
                path = os.path.join(path, figures_dirname)
                if not os.path.isdir(path):
                    os.mkdir(path=path)
                pylab.savefig(fname=os.path.join(path, 'HiddenTier{}'.format(str(hidden_tier))))
            pylab.close('all')
