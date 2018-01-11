import nest
from memristive_spinal_cord.layer1.neuron_network import NeuronNetwork
import memristive_spinal_cord.util as util


from memristive_spinal_cord.layer1.moraud.params.original.afferents import afferent_params
from memristive_spinal_cord.layer1.moraud.params.original.neuron_groups import neuron_group_params
from memristive_spinal_cord.layer1.moraud.params.original.devices import device_params
from memristive_spinal_cord.layer1.moraud.params.original.connections import connection_params_list

entity_params = dict()
entity_params.update(afferent_params)
entity_params.update(neuron_group_params)
entity_params.update(device_params)

util.clean_previous_results()
layer1 = NeuronNetwork(entity_params, connection_params_list)

nest.Simulate(100.)
