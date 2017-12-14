from memristive_spinal_cord.layer1.neuron_network import NeuronNetwork
from memristive_spinal_cord.layer1.moraud.params.neucogar import entities, connectome

layer1 = NeuronNetwork(entities.params_storage, connectome.params_storage)
