from neucogar.api_kernel import CreateNetwork
from memristive_spinal_cord.layer1.neuron_group_network import NeuronGroupNetwork
import memristive_spinal_cord.layer1.rybak.params.neucogar as layer1_params

network = NeuronGroupNetwork()

r_motor = network.create_neuron_group("R Motoneurons", layer1_params.r_motor_params)
l_motor = network.create_neuron_group("L Motoneurons", layer1_params.l_motor_params)

r_renshaw = network.create_neuron_group("R Renshaw", layer1_params.r_renshaw_params)
l_renshaw = network.create_neuron_group("L Renshaw", layer1_params.l_renshaw_params)

r_inter_1a = network.create_neuron_group("R 1A Interneurons", layer1_params.r_inter_1a_params)
l_inter_1a = network.create_neuron_group("L 1A Interneurons", layer1_params.l_inter_1a_params)

r_inter_1b = network.create_neuron_group("R 1B Interneurons", layer1_params.r_inter_1b_params)
l_inter_1b = network.create_neuron_group("L 1B Interneurons", layer1_params.l_inter_1b_params)

CreateNetwork(network.get_neuron_number())
