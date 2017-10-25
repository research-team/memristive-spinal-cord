import pylab
import nest

print ("Creating neuron.")
neuron = nest.Create("iaf_psc_alpha")
print(nest.GetStatus(neuron, "I_e"))
print(nest.GetStatus(neuron, ["V_reset", "V_th"]))
multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})
spikedetector = nest.Create("spike_detector",
                params={"withgid": True, "withtime": True})

nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)

print ("Started simulation")

nest.Simulate(1000.0)

print ("Simulation done.")