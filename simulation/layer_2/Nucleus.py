__author__ = "max talanov"

import logging.config
from func import *


class Nucleus:
    """
    The class of neuronal nucleus
    """

    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('Nucleus_logger')

    name = ""
    number_of_neurons = 0
    neurons = []
    neuron_model =""
    synapse_model = ""
    multimeters = []

    def __init__(self, name):
        self.name = name

    def __init__(self, name, neuron_model, number_of_neurons):
        self.name = name
        self.neuron_model = neuron_model
        self.number_of_neurons = number_of_neurons
        self.neurons = self.generate_nucleus(self.neuron_model, self.number_of_neurons)

    def __str__(self):
        return "Nucleus:" + self.name + ", NN=" + str(self.number_of_neurons)

    def generate_nucleus(self, neuron_model, neurons_per_nucleus):
        """
        Generates the nucleus of neurons with specified neuronal model and number of neurons.
        :param neuron_model: the neuronal modal to use
        :param neurons_per_nucleus: the number of neurons to generate
        :return: the list of neurons generated
        """
        self.logger.debug("Generating %s cells", neurons_per_nucleus)
        res = nest.Create(neuron_model, neurons_per_nucleus)
        self.logger.debug(res)
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
        self.logger.debug("{0} -> {1} ({2}) w[{3}] // "
                     "{4}x{5}={6} synapses".format(self.name, post.name, syn_type[:-8], weight, self.number_of_neurons,
                                                   MaxSynapses if post.number_of_neurons > MaxSynapses else post.number_of_neurons,
                                                   connections))

    def connect_multimeter(self):
        """
        Creates multimeter and connects to all neurons of the nucleus
        :return:
        """
        name = self.name
        self.multimeters = nest.Create('multimeter', params=multimeter_param)  #todo: add count of multimeters
        nest.Connect(self.multimeters, (self.neurons[:N_volt]))
        logger.debug("Multimeter => {0}. On {1}".format(name, self.neurons[:N_volt]))