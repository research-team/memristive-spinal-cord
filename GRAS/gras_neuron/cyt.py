from Interface import *

a = Py_Group('simple_neuron')
b = Py_Group('simple_dimple')
gen = py_add_gen()
py_connect_ind(a, b, 0.025, 0.03)


py_simulate()