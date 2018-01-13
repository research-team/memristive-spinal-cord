from neucogar.Nucleus import Nucleus
from neucogar.api_kernel import CreateNetwork

from memristive_spinal_cord.layer2.models import Neurotransmitters, ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk3.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk3.components.parameters import TierWeights
from memristive_spinal_cord.layer2.schemes.mk3.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.mk3.polysynaptic_circuit import PolysynapticCircuit


class Layer2:
    def __init__(self):
        self.polysynaptic_circuit = PolysynapticCircuit()
        self.mediator = Nucleus(nucleus_name='Mediator')
        self.interneuronal_pool = Nucleus(nucleus_name='InterneuronalPool')

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

        CreateNetwork(simulation_neuron_number=82)
        self.set_connections()
        self.connect_multimeters()
        self.connect_noise_generator()

    def connect_noise_generator(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).ConnectPoissonGenerator(
            weight=TierWeights.SG.value,
            start=1,
            stop=10,
            rate=100
        )

    def connect_multimeters(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.interneuronal_pool.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        for tier in self.polysynaptic_circuit.get_tiers():
            tier.connect_multimeters()

    def set_connections(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.polysynaptic_circuit.get_input().nuclei(
                Neurotransmitters.GLU.value
            ),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=TierWeights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )
        for tier in self.polysynaptic_circuit.get_tiers():
            tier.set_connections()