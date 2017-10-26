import pylab
import nest
import logging
#logging.basicConfig(filename='example.log',level=logging.DEBUG)

#hh_psc_alpha, hh_psc_alpha_gap

neuron = nest.Create("hh_psc_alpha")
neuron2 = nest.Create("hh_psc_alpha_gap")

logging.debug("Setting parameters of neurons")
nest.SetStatus(neuron2 , {"I_e": 370.0})


logging.debug("Creating devices")
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
spikedetector = nest.Create("spike_detector",
                params={"withgid": True, "withtime": True})

nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)
nest.Connect(multimeter, neuron2)

print ("Started simulation")

nest.Simulate(1000.0)

dmm = nest.GetStatus(multimeter)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
pylab.figure(1)
pylab.plot(ts, Vms)
dSD = nest.GetStatus(spikedetector,keys="events")[0]
evs = dSD["senders"]
ts = dSD["times"]

pylab.figure(2)
Vms1 = dmm["events"]["V_m"][::2] # start at index 0: till the end: each second entry
ts1 = dmm["events"]["times"][::2]
pylab.plot(ts1, Vms1)
Vms2 = dmm["events"]["V_m"][1::2] # start at index 1: till the end: each second entry
ts2 = dmm["events"]["times"][1::2]
pylab.plot(ts2, Vms2)

pylab.show()


print ("Simulation done.")