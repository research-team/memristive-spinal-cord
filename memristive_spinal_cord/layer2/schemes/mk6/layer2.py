from neucogar.api_kernel import CreateNetwork

from memristive_spinal_cord.layer2.schemes.mk6.polysynaptic_circuit import PolysynapticCircuit
from memristive_spinal_cord.layer2.schemes.mk6.terminals import Terminals


class Layer2:

    def __init__(self):
        self.pc = PolysynapticCircuit()
        self.terminals = Terminals()
        CreateNetwork(simulation_neuron_number=self.terminals.get_number_of_neurons() + self.pc.get_number_of_neurons())
        self.set_interconnections()
        self.connect_multimeters()
        self.terminals.connect_spike_generator()

    def set_interconnections(self):
        self.pc.set_interconnections()
        self.terminals.connect(
            input=self.pc.get_input(),
            output=self.pc.get_output()
        )

    def connect_multimeters(self):
        self.pc.connect_multimeters()
        self.terminals.connect_multimeters()
