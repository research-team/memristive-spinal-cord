__author__ = "max talanov"

import logging.config
from memristive_spinal_cord.layer_2.func import *


class Nucleus:
    """
    The class of neuronal nucleus
    """

    # logger
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('Nucleus_logger')
    # defaults
    pg_delay = 5.
    min_neurons = 10
    max_synapses = 99999
    min_synapses = 10
    """ Devices """
    # Neurons number for spike detector
    N_detect = 100

    # Neurons number for multimeter
    N_volt = 3

    name = ""
    number_of_neurons = 0
    neurons = []
    neuron_model =""
    synapse_model = ""
    multimeters = []
    spike_detectors = []
    generator = []

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
                     'multapses': True,
                     'autapses': False}
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

    def connect_detector(self):
        """
        Creates spike detectors and connects them to all neurons of the nucleus
        :return: spike detectors
        """
        name = self.name
        # Init number of neurons which will be under detector watching
        number = self.number_of_neurons if self.number_of_neurons < N_detect else N_detect
        # Add to spikeDetectors a new detector
        self.spike_detectors = nest.Create('spike_detector', params=detector_param)
        # Connect N first neurons ID of part with detector
        nest.Connect(self.neurons[:number], self.spike_detectors)
        # Show data of new detector
        logger.debug("Detector => {0}. Tracing {1} neurons".format(name, number))
        return self.spike_detectors

    def connect_Poisson_generator(self, start=1, stop=50, rate=250, prob=1., weight=1000):
        """
        The poisson_generator simulates a neuron that is firing with Poisson statistics, i.e. exponentially
        distributed interspike intervals. It will generate a _unique_ spike train for each of it's targets.

        :param start: double the time of the simulation when the generator starts working with respect to origin in ms
        :type stop: double
        :param stop: the simulation time when the device stops working with respect to origin in ms
        :type stop: double
        :param rate: mean firing rate in Hz
        :type rate: double
        :param prob: the connection probability
        :type prob: double
        :param weight: the strength of a signal in nS
        :type weight: double
        :return: the generator
        """

        outdegree = int(self.number_of_neurons * prob)
        self.generator = nest.Create('poisson_generator', 1, {'rate': float(rate),
                                                              'start': float(start),
                                                              'stop': float(stop)})
        conn_spec = {'rule': 'fixed_outdegree',
                     'outdegree': outdegree}
        syn_spec = {
            'weight': float(weight),
            'delay': float(self.pg_delay)}
        nest.Connect(self.generator, self.neurons, conn_spec=conn_spec, syn_spec=syn_spec)
        logger.info("(ID:{0}) to {1} ({2}/{3}). Interval: {4}-{5}ms".format(
            self.generator[0],
            self.name,
            outdegree,
            self.number_of_neurons,
            start,
            stop
        ))
        return self.generator

    def connect_spike_detector(self, neurons_connected_to_detector = N_detect):
        """
        Connects spike detector to the  neuron of the nucleus

        :param neurons_connected_to_detector: the number of neurons to be connected to the detector
        :type neurons_connected_to_detector: int
        :return: detector
        """

        name = self.name
        detector_param = {'label': name,
                          'withgid': True,
                          'to_file': False,
                          'to_memory': True}  # withweight true

        number = self.number_of_neurons if self.number_of_neurons < neurons_connected_to_detector else neurons_connected_to_detector
        tracing_ids = self.neurons[:number]
        self.spike_detectors = nest.Create('spike_detector', params=detector_param)
        nest.Connect(tracing_ids, self.spike_detectors)
        logger.info("(ID:{0}) to {1} ({2}/{3})".format(self.spike_detectors[0], name, len(tracing_ids), self.number_of_neurons))
        return self.spike_detectors

    def connect_multimeter(self):
        """
        Creates multimeters and connects them to every neuron of the nucleus.
        :return: multimeter
        """
        name = self.name
        multimeter_param = {'label': name,
                            'withgid': True,
                            'withtime': True,
                            'to_file': False,
                            'to_memory': True,
                            'interval': 0.1,
                            'record_from': ['V_m']}
        tracing_ids = self.neurons[:N_volt]
        self.multimeters = nest.Create('multimeter', params=multimeter_param)  # ToDo add count of multimeters
        nest.Connect(self.multimeters, tracing_ids)
        logger.info("(ID:{0}) to {1} ({2}/{3})".format(self.multimeters[0], name, len(tracing_ids), self.number_of_neurons))
        return self.multimeters
