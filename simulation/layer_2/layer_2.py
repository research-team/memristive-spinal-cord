import pylab
import nest
import logging
import logging.config
from func import *

neuron_model = "hh_psc_alpha_gap"

def generate_neurons(neurons_per_nucleus, n_of_projections, n_of_layers):
    part = [n_of_layers]
    for part_id in range(0, n_of_layers):
        part[part_id] = nest.Create(neuron_model, neurons_per_nucleus)
        logger.debug("{0} [{1}] neurons".format(part[part_id], neurons_per_nucleus))
    return part


logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('layer_2_logger')

logger.info("START")

#hh_psc_alpha, hh_psc_alpha_gap

logger.debug("Creating neurons")
neuron = nest.Create("hh_psc_alpha")
neuron2 = nest.Create("hh_psc_alpha_gap")

neurons = generate_neurons(20, 200, 1)
logger.debug("neurons", neurons)

logger.debug("Creating synapses")
nest.CopyModel('stdp_synapse', glu_synapse, STDP_synparams_Glu)
nest.CopyModel('stdp_synapse', gaba_synapse, STDP_synparams_GABA)

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

nest.Simulate(1000.0)

logger.debug("Graphs")
dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

pylab.figure(1)
pylab.plot(ts, Vms)
dSD = nest.GetStatus(spikedetector, keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]
pylab.figure(2)
#pylab.plot(evs, Vms)

pylab.figure(3)
Vms1 = dmm["events"]["V_m"][::2] # start at index 0: till the end: each second entry
ts1 = dmm["events"]["times"][::2]
pylab.plot(ts1, Vms1)
Vms2 = dmm["events"]["V_m"][1::2] # start at index 1: till the end: each second entry
ts2 = dmm["events"]["times"][1::2]
pylab.plot(ts2, Vms2)

pylab.show()

logger.info("Simulation done.")