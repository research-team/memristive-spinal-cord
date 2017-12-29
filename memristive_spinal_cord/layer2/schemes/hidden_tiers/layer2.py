from memristive_spinal_cord.layer2.schemes.hidden_tiers.polysynaptic_circuit import PolysynapticCircuit
from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Weights
from memristive_spinal_cord.layer2.models import ConnectionTypes


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

    def set_connections(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.polysynaptic_circuit.get_input().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )
        for tier, weight in zip(self.polysynaptic_circuit.get_output(), Weights.LIP.value.reverse()):
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

    def connect_multimeters(self):
        self.mediator.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.interneuronal_pool.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        for tier in self.polysynaptic_circuit.get_tiers():
            tier.connect_multimeters()
        for hidden_tier in self.polysynaptic_circuit.get_hidden_tiers():
            hidden_tier.connect_multimeters()

