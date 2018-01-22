from neucogar.api_kernel import CreateNetwork

from memristive_spinal_cord.layer2.schemes.mk5.polysynaptic_circuit import PolysynapticCircuit
from memristive_spinal_cord.layer2.schemes.mk5.terminals import Terminals


class Layer2:
    def __init__(self):
        self.polysynaptic_circuit = PolysynapticCircuit()
        self.terminals = Terminals()

        CreateNetwork(
            simulation_neuron_number=PolysynapticCircuit.get_number_of_neurons() + Terminals.get_number_of_neurons()
        )
        self.set_connections()
        self.terminals.connect_spike_generator()
        self.connect_multimeters()

    def set_connections(self):
        self.polysynaptic_circuit.set_connections()
        self.terminals.connect(
            input=self.polysynaptic_circuit.get_input(),
            output=self.polysynaptic_circuit.get_output()
        )

    def connect_multimeters(self):
        self.polysynaptic_circuit.connect_multimeters()
        self.terminals.connect_multimeters()
