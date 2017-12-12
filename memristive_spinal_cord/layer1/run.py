from neucogar.api_kernel import CreateNetwork
from memristive_spinal_cord.layer1.neuron_network import NeuronNetwork
from memristive_spinal_cord.layer1.moraud.neuron_group_names import Layer1NeuronGroupNames
from memristive_spinal_cord.layer1.moraud.params.neucogar.neuron_groups import layer1_neuron_group_params
import memristive_spinal_cord.layer1.moraud.params.neucogar.connectome as connectome_params

layer1_network = NeuronNetwork(layer1_neuron_group_params)

for group_name in Layer1NeuronGroupNames:
    layer1_network.create_neuron_group(
        group_name,
        layer1_neuron_group_params[group_name]
    )

CreateNetwork(layer1_network.get_neuron_number())

connectome_params.connect(layer1_network)