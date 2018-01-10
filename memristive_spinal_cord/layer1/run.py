import neucogar.api_kernel as neucogar_api
import neucogar.api_diagrams as neucogar_diagrams
import definitions
from memristive_spinal_cord.layer1.neuron_network import NeuronNetwork
from memristive_spinal_cord.layer1.moraud.params.original import entities, connectome
import memristive_spinal_cord.util as util

util.clean_previous_results()
layer1 = NeuronNetwork(entities.params_storage, connectome.params_storage)


import memristive_spinal_cord.layer1.params.device_params as device_params
from memristive_spinal_cord.layer1.moraud.neuron_groups import Layer1Neurons
from memristive_spinal_cord.layer1.moraud.devices import Layer1Devices
import pylab

neurons = layer1.get_entity(Layer1Neurons.FLEX_MOTOR)
flex_inter_1a_multimeter = neucogar_api.NEST.Create(
    'multimeter',
    params={"record_from": ["V_m"], "withtime": True, "interval": 0.1}
)
neucogar_api.NEST.Connect(flex_inter_1a_multimeter, neurons)


def plot_parameter(plot_name, device, param_to_display, bla=None):
    status = neucogar_api.NEST.GetStatus(device)[0]
    events = status['events']
    times = events['times']
    pylab.figure(plot_name)
    if bla:
        pylab.plot(times, events[param_to_display], bla)
    else:
        pylab.plot(times, events[param_to_display])



neucogar_api.NEST.Simulate(200.)

plot_parameter('V_m', flex_inter_1a_multimeter, 'V_m')
pylab.show()
# neucogar_diagrams.BuildVoltageDiagrams(definitions.RESULTS_DIR)

