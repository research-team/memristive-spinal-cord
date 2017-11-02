__author__ = "max talanov"

import pylab
import nest
import logging
import logging.config
from func import *
import property
from Nucleus import Nucleus

neuron_model = "hh_psc_alpha_gap"
number_of_layers = 6

#Kernel setup
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 8})

def generate_layers(neuron_model, neurons_per_nucleus, n_of_layers, weight_ex, weight_in):
    """
    Generate neuronal layers with specified topology, see https://github.com/research-team/memristive-spinal-cord
    :param neuron_model: the neuron model to use
    :param neurons_per_nucleus: the number of neurons per nucleus to be generated
    :param n_of_layers: the number of layers
    :param weight_ex: the excitatory weight
    :param weight_in: the inhibitory weight
    :return: the list of 6 layers of dictionary of {"left","right","inhibitory"} of connected neurons
    """
    layers = []
    for i in range(0, n_of_layers):
        logger.debug("Generating %s layer", i)
        # creating nuclei

        nucleus_left = Nucleus("left", neuron_model, neurons_per_nucleus)
        nucleus_right = Nucleus("right", neuron_model, neurons_per_nucleus)
        # connecting nuclei
        nucleus_left.connect(nucleus_right, syn_type=Glu, weight_coef=weight_ex)
        nucleus_right.connect(nucleus_left, syn_type=Glu, weight_coef=weight_in)
        if (i>0):
            prev_layer = layers[i-1]
            # inhibitory nucleus
            nucleus_inhibitory = Nucleus("inhibitory", neuron_model, neurons_per_nucleus)
            nucleus_right.connect(nucleus_inhibitory, syn_type=GABA, weight_coef=weight_ex)
            nucleus_inhibitory.connect(prev_layer["right"])
            # from right nucleus of previous layer to the current layer
            nucleus_right.connect(prev_layer["right"], syn_type=Glu, weight_coef=weight_ex)
            # from left nucleus of the previous to the left nucleus of the current layer
            prev_layer["left"].connect(nucleus_left, syn_type=Glu, weight_coef=weight_ex)

            layers.append({"left": nucleus_left, "right": nucleus_right, "inh": nucleus_inhibitory})
        else:
            layers.append({"left": nucleus_left, "right": nucleus_right})
        logger.debug("Packing %s layer in list", i)
    return layers

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('layer_2_logger')
logger.info("START")

logger.debug("Creating neurons")
#hh_psc_alpha, hh_psc_alpha_gap
#test neurons

logger.debug("Creating synapses")
nest.CopyModel('stdp_synapse', glu_synapse, STDP_synparams_Glu)
nest.CopyModel('stdp_synapse', gaba_synapse, STDP_synparams_GABA)

layers = generate_layers(neuron_model, 20, number_of_layers, 0.2, 0.3)
logger.debug("Layers created %s", len(layers))


logger.debug("Creating devices")
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})

logger.debug("Connecting")
# generators
layers[0]["right"].connect_Poisson_generator()
in_multimeter = layers[0]["right"].connect_multimeter()


# out of layers
for i in range(0, number_of_layers):
    layers[i]["left"].connect_multimeter()
    layers[i]["left"].connect_detector()


#nest.Connect(neuron, spikedetector)

logger.debug("Started simulation")

nest.Simulate(1000.0)
logger.debug("Simulation done.")

logger.debug("Graphs")

dmm = nest.GetStatus(layers[0]["right"].multimeters)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
pylab.figure("layer 1 right")
pylab.plot(ts, Vms)

dmm = nest.GetStatus(layers[0]["left"].multimeters)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
pylab.figure("layer 1 left")
pylab.plot(ts, Vms)

dmm = nest.GetStatus(layers[1]["left"].multimeters)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
pylab.figure("layer 2 left")
pylab.plot(ts, Vms)

dmm = nest.GetStatus(layers[2]["left"].multimeters)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
pylab.figure("layer 3 left")
pylab.plot(ts, Vms)


#dSD = nest.GetStatus(spikedetector, keys="events")[0]
#evs = dSD["senders"]
#ts = dSD["times"]
#pylab.figure(2)
#pylab.plot(evs, Vms)
#
# pylab.figure(3)
# Vms1 = dmm["events"]["V_m"][::2] # start at index 0: till the end: each second entry
# ts1 = dmm["events"]["times"][::2]
# pylab.plot(ts1, Vms1)
# Vms2 = dmm["events"]["V_m"][1::2] # start at index 1: till the end: each second entry
# ts2 = dmm["events"]["times"][1::2]
# pylab.plot(ts2, Vms2)
#
pylab.show()

logger.info("DONE.")