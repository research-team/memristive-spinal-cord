__author__ = "max talanov"

import pylab
import nest
import logging
import logging.config
from func import *
import property

neuron_model = "hh_psc_alpha_gap"


def generate_nucleus(neuron_model, neurons_per_nucleus):
    """
    Generates the nucleus of neurons with specified neuronal model and number of neurons.
    :param neuron_model: the neuronal modal to use
    :param neurons_per_nucleus: the number of neurons to generate
    :return: the list of neurons generated
    """
    logger.debug("Generating %s cells", neurons_per_nucleus)
    res = nest.Create(neuron_model, neurons_per_nucleus)
    logger.debug(res)
    return res


def generate_layers(neuron_model, neurons_per_nucleus, n_of_projections, n_of_layers, weight_ex, weight_in):
    """
    Generate neuronal layers with specified topology, see https://github.com/research-team/memristive-spinal-cord
    :param neuron_model: the neuron model to use
    :param neurons_per_nucleus: the number of neurons per nucleus to be generated
    :param n_of_projections: the number of connections from one nucleus to another according to topology
    :param n_of_layers: the number of layers
    :param weight_ex: the excitatory weight
    :param weight_in: the inhibitory weight
    :return: the list of list of connected neurons
    """
    layers = []
    for i in range(0, n_of_layers):
        logger.debug("Generating %s layer", i)
        # creating nuclei
        nucleus_left = generate_nucleus(neuron_model, neurons_per_nucleus)
        nucleus_right = generate_nucleus(neuron_model, neurons_per_nucleus)
        # connecting nuclei
        connect(nucleus_left, nucleus_right, syn_type=Glu, weight_coef=weight_ex)
        connect(nucleus_left, nucleus_right, syn_type=Glu, weight_coef=weight_in)

        #todo: add connection to previous layer
        if (i>0):
            nucleus_inhibitory = generate_nucleus(neuron_model, neurons_per_nucleus)
            connect(nucleus_right, nucleus_inhibitory, syn_type=GABA, weight_coef=weight_ex)
            connect(nucleus_inhibitory, layers[i-1][1])
            layers.append([nucleus_left, nucleus_right, nucleus_inhibitory])
        else:
            layers.append([nucleus_left, nucleus_right])
        logger.debug("Packing %s layer in list", i)
    return layers

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('layer_2_logger')
logger.info("START")

logger.debug("Creating neurons")
#hh_psc_alpha, hh_psc_alpha_gap
#test neurons
neuron = nest.Create("hh_psc_alpha")
neuron2 = nest.Create("hh_psc_alpha_gap")

logger.debug("Creating synapses")
nest.CopyModel('stdp_synapse', glu_synapse, STDP_synparams_Glu)
nest.CopyModel('stdp_synapse', gaba_synapse, STDP_synparams_GABA)

layers = generate_layers(neuron_model, 20, 200, 6, 0.2, 0.3 )
logger.debug("Layers created %s", len(layers))


logger.debug("Setting parameters of neurons")
nest.SetStatus(neuron2 , {"I_e": 370.0})

logger.debug("Creating devices")
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
spikedetector = nest.Create("spike_detector", params={"withgid": True, "withtime": True})

logger.debug("Connecting")
nest.Connect(neuron, neuron2)
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)
nest.Connect(multimeter, neuron2)

logger.debug ("Started simulation")

#nest.Simulate(1000.0)

# logger.debug("Graphs")
# dmm = nest.GetStatus(multimeter)[0]
# Vms = dmm["events"]["V_m"]
# ts = dmm["events"]["times"]
#
# pylab.figure(1)
# pylab.plot(ts, Vms)
# dSD = nest.GetStatus(spikedetector, keys="events")[0]
# evs = dSD["senders"]
# ts = dSD["times"]
# pylab.figure(2)
# #pylab.plot(evs, Vms)
#
# pylab.figure(3)
# Vms1 = dmm["events"]["V_m"][::2] # start at index 0: till the end: each second entry
# ts1 = dmm["events"]["times"][::2]
# pylab.plot(ts1, Vms1)
# Vms2 = dmm["events"]["V_m"][1::2] # start at index 1: till the end: each second entry
# ts2 = dmm["events"]["times"][1::2]
# pylab.plot(ts2, Vms2)
#
# pylab.show()

logger.info("Simulation DONE.")