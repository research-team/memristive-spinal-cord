from neuron import h
h.load_file('nrngui.hoc')

#paralleling NEURON staff
pc = h.ParallelContext()
rank = int(pc.id()) 
nhost = int(pc.nhost())

print(f"I am {pc.id} of {pc.nhost}")
pc.runworker()
pc.done()
h.quit()