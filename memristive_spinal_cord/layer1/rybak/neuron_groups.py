from neucogar.api_kernel import CreateNetwork
from memristive_spinal_cord.layer1.neuron_group_network import NeuronGroupNetwork
from memristive_spinal_cord.layer1.rybak.params.neucogar import Layer1Groups

layer1_network = NeuronGroupNetwork()

for group in Layer1Groups:
    layer1_network.create_neuron_group(group)

CreateNetwork(layer1_network.get_neuron_number())
