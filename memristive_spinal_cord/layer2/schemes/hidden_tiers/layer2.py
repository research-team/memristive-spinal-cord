from memristive_spinal_cord.layer2.schemes.hidden_tiers.polysynaptic_circuit import PolysynapticCircuit
from neucogar.Nucleus import Nucleus
from neucogar.api_kernel import CreateNetwork
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Weights
from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Constants
import neucogar.api_kernel as api_kernel


class Layer2:
    def __init__(self):
        self.polysynaptic_circuit = PolysynapticCircuit()
        self.mediator = Nucleus('Mediator')
        self.interneuronal_pool = Nucleus('InterneuronalPool')

        self.mediator.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )

        self.interneuronal_pool.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )

        CreateNetwork(simulation_neuron_number=PolysynapticCircuit.get_number_of_neurons() + 2)
        self.set_connections()
        self.connect_multimeters()
        self.connect_spike_generator()

    def set_connections(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.polysynaptic_circuit.get_input().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )
        for tier, weight in zip(self.polysynaptic_circuit.get_output(), Weights.LIP.value[::-1]):
            tier.nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.interneuronal_pool.nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=weight,
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )
        for tier in self.polysynaptic_circuit.get_tiers():
            tier.set_connections()
        for hidden_tier in self.polysynaptic_circuit.get_hidden_tiers():
            hidden_tier.set_connections()

        self.polysynaptic_circuit.set_connections()

    def connect_multimeters(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.interneuronal_pool.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        for tier in self.polysynaptic_circuit.get_tiers():
            tier.connect_multimeters()
        for hidden_tier in self.polysynaptic_circuit.get_hidden_tiers():
            hidden_tier.connect_multimeters()

    def connect_spike_generator(self):
        spike_generator = api_kernel.NEST.Create('spike_generator', 1, {
            'spike_times': Constants.SPIKE_GENERATOR_TIMES.value,
            'spike_weights': Constants.SPIKE_GENERATOR_WEIGHTS.value
        })

        api_kernel.NEST.Connect(
            spike_generator,
            self.mediator.nuclei(Neurotransmitters.GLU.value).getNeurons())

