import neucogar.api_kernel as neucogar_api
import neucogar.api_diagrams as neucogar_diagrams
import definitions
from memristive_spinal_cord.layer1.neuron_network import NeuronNetwork
from memristive_spinal_cord.layer1.moraud.params.neucogar import entities, connectome
import memristive_spinal_cord.util as util


def connect_devices(layer1):
    from memristive_spinal_cord.layer1.moraud.devices import Layer1Devices
    from memristive_spinal_cord.layer1.moraud.neuron_groups import Layer1Neurons

    multimeter = layer1.get_entity(Layer1Devices.FLEX_INTER_1A_MULTIMETER)
    neurons = layer1.get_entity(Layer1Neurons.FLEX_INTER_1A)
    neucogar_api.NEST.Connect(multimeter, neurons)

    detector = layer1.get_entity(Layer1Devices.FLEX_INTER_1A_DETECTOR)
    neurons = layer1.get_entity(Layer1Neurons.FLEX_INTER_1A)
    neucogar_api.NEST.Connect(neurons, detector)


util.clean_previous_results()
layer1 = NeuronNetwork(entities.params_storage, connectome.params_storage)
connect_devices(layer1)
neucogar_api.NEST.Simulate(100.)
neucogar_diagrams.BuildVoltageDiagrams(definitions.RESULTS_DIR)
