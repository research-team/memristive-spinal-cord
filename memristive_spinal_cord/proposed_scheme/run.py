import nest
import definitions
from memristive_spinal_cord.proposed_scheme.neuron_network import NeuronNetwork
import memristive_spinal_cord.util as util
import shutil
import os

from memristive_spinal_cord.proposed_scheme.moraud.params.original.afferents import afferent_params
from memristive_spinal_cord.proposed_scheme.moraud.params.original.neuron_groups import neuron_group_params
from memristive_spinal_cord.proposed_scheme.moraud.params.original.devices import device_params
from memristive_spinal_cord.proposed_scheme.moraud.params.original.connections import connection_params_list
from memristive_spinal_cord.proposed_scheme.moraud.entities import Layer1Multimeters
from memristive_spinal_cord.proposed_scheme.results_plotter import ResultsPlotter
import memristive_spinal_cord.proposed_scheme.device_data as device_data

from memristive_spinal_cord.proposed_scheme.level2.devices import l2_device_params
from memristive_spinal_cord.proposed_scheme.level2.neuron_groups import l2_neuron_group_params
from memristive_spinal_cord.proposed_scheme.level2.connections import l2_connections_list

from memristive_spinal_cord.proposed_scheme.level2.parameters import SIMULATION_TIME

nest.SetKernelStatus({"total_num_virtual_procs": 8,
                      "print_time": True})


def plot_neuron_groups(flexor_device, extensor_device, group_name: str) -> None:

    flexor_motor_data = device_data.get_average_voltage(
        flexor_device,
        definitions.RESULTS_DIR
    )
    extensor_motor_data = device_data.get_average_voltage(
        extensor_device,
        definitions.RESULTS_DIR
    )
    plotter.subplot(flexor_motor_data, extensor_motor_data, group_name)


def plot_neuron_group(multimeter_device, group_name: str) -> None:

    data = device_data.get_average_voltage(
        multimeter_device,
        definitions.RESULTS_DIR
    )
    plotter.subplot_one_figure(data, group_name)


nest.Install("research_team_models")

entity_params = dict()
entity_params.update(afferent_params)
entity_params.update(neuron_group_params)
entity_params.update(device_params)
entity_params.update(l2_neuron_group_params)
entity_params.update(l2_device_params)
connection_params_list += l2_connections_list

util.clean_previous_results()
layer1 = NeuronNetwork(entity_params, connection_params_list)

nest.Simulate(SIMULATION_TIME)

img_path = os.path.join(definitions.RESULTS_DIR, 'img')
if os.path.isdir(img_path):
    shutil.rmtree(img_path)
os.mkdir(img_path)

plotter = ResultsPlotter(4, 'Layer1 average "V_m" of neuron groups')
plotter.reset()
plot_neuron_groups(Layer1Multimeters.FLEX_MOTOR, Layer1Multimeters.EXTENS_MOTOR, 'Motor')
plot_neuron_groups(Layer1Multimeters.FLEX_INTER_2, Layer1Multimeters.EXTENS_INTER_2, 'Inter2')
plot_neuron_groups(Layer1Multimeters.FLEX_INTER_1A, Layer1Multimeters.EXTENS_INTER_1A, 'Inter1A')
plot_neuron_groups('Pool1-multimeter', 'Pool0-multimeter', 'Pool')
plotter.save(os.path.join(img_path, 'pool-level1.png'))

for tier in range(6, 0, -1):
    plotter = ResultsPlotter(7, 'Tier{}'.format(tier))
    plotter.reset()
    for exc in range(5):
        plot_neuron_group('Tier{}E{}-multimeter'.format(tier, exc), 'Tier{}E{}'.format(tier, exc))
    plot_neuron_group('Tier{}I0-multimeter'.format(tier), 'Tier{}I0'.format(tier))
    plot_neuron_group('Tier{}I1-multimeter'.format(tier), 'Tier{}I1'.format(tier))
    plotter.save(os.path.join(img_path, 'level2-tier{}.png'.format(tier)))

plotter = ResultsPlotter(3, 'Tier0')
plotter.reset()
for exc in range(2):
    plot_neuron_group('Tier0E{}-multimeter'.format(exc), 'Tier0E{}'.format(exc))
plot_neuron_group('Tier0I0-multimeter', 'Tier0I0')
plotter.save(os.path.join(img_path, 'level2-tier0.png'))

with open(os.path.join(img_path, 'results.md'), 'w') as result_file:
    for tier in range(6, -1, -1):
        result_file.write('**Tier{}**\n'.format(tier))
        result_file.write('![Tier{}](level2-tier{}.png)\n'.format(tier, tier))
    result_file.write('**Pool-Level1**\n')
    result_file.write('![Pool-Level1](pool-level1.png)\n')
