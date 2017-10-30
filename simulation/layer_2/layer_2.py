__author__ = "max talanov"

import pylab
import nest
import logging
import logging.config
from func import *
import property

neuron_model = "hh_psc_alpha_gap"

class Nucleus:
    """
    The class of neuronal nucleus
    """
    name = ""
    number_of_neurons = 0
    neurons = []
    neuron_model =""
    synapse_model = ""

    def __init__(self, name):
        self.name = name

    def __init__(self, name, neuron_model, number_of_neurons):
        self.name = name
        self.neuron_model = neuron_model
        self.number_of_neurons = number_of_neurons
        self.neurons = self.generate_nucleus(self.neuron_model, self.number_of_neurons)

    def generate_nucleus(self, neuron_model, neurons_per_nucleus):
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

    def connect(pre_synaptic_nucleus, post_synaptic_nucleus, syn_type=GABA, weight_coef=1):
        """
        :param pre_synaptic_nucleus: the presynaptic nucleus
        :type pre_synaptic_nucleus: Nucleus
        :param post_synaptic_nucleus: the postsynaptic nucleus
        :param pre_synaptic_nucleus: Nucleus
        :param syn_type: int type of synapses from the synapses dictionary see property.py
        :param weight_coef: float weight of synapses
        :return:
        """
        # Set new weight value (weight_coef * basic weight)
        nest.SetDefaults(synapses[syn_type][model], {'weight': weight_coef * synapses[syn_type][basic_weight]})
        # Create dictionary of connection rules
        conn_dict = {'rule': 'fixed_outdegree',
                     'outdegree': MaxSynapses if post_synaptic_nucleus.number_of_neurons > MaxSynapses
                     else post_synaptic_nucleus.number_of_neurons,
                     'multapses': False}
        # Connect PRE IDs neurons with POST IDs neurons, add Connection and Synapse specification
        nest.Connect(pre_synaptic_nucleus.neurons, post_synaptic_nucleus.neurons, conn_spec=conn_dict, syn_spec=synapses[syn_type][model])
        # Show data of new connection
        pre_synaptic_nucleus.log_connection(post_synaptic_nucleus, synapses[syn_type][model], nest.GetDefaults(synapses[syn_type][model])['weight'])

    def log_connection(self, post, syn_type, weight):
        """
        Logging the synaptic connection
        :param self: presynaptic nuleus
        :param post: postsynaptic nucleus
        :type past: Nucleus
        :param syn_type: string the type of synapse
        :param weight: float the synaptic weight of connection
        :return:
        """
        global SYNAPSES
        connections = self.number_of_neurons * post.number_of_neurons if post.number_of_neurons < MaxSynapses else self.number_of_neurons * MaxSynapses
        SYNAPSES += connections
        logger.debug("{0} -> {1} ({2}) w[{3}] // "
                     "{4}x{5}={6} synapses".format(self.name, post.name, syn_type[:-8], weight, self.number_of_neurons,
                                                   MaxSynapses if post.number_of_neurons > MaxSynapses else post.number_of_neurons,
                                                   connections))

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

        nucleus_left = Nucleus("left", neuron_model, neurons_per_nucleus)
        nucleus_right = Nucleus("right", neuron_model, neurons_per_nucleus)
        # connecting nuclei
        nucleus_left.connect(nucleus_right, syn_type=Glu, weight_coef=weight_ex)
        nucleus_left.connect(nucleus_right, syn_type=Glu, weight_coef=weight_in)

        #todo: add connection to previous layer
        if (i>0):
            nucleus_inhibitory = Nucleus("inhibitory", neuron_model, neurons_per_nucleus)
            nucleus_right.connect(nucleus_inhibitory, syn_type=GABA, weight_coef=weight_ex)
            nucleus_inhibitory.connect(layers[i-1]["right"])
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
neuron = nest.Create("hh_psc_alpha")
neuron2 = nest.Create("hh_psc_alpha_gap")

layer1 = ({k_name: 'Layer 1 [Glu0]', k_NN: 20},{k_name: 'Layer 1 [Glu1]', k_NN: 20}, {k_name: "Layer 1 [GABA]", k_NN: 20})

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