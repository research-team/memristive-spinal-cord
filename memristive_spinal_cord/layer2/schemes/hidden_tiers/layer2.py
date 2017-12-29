from memristive_spinal_cord.layer2.schemes.hidden_tiers.polysynaptic_circuit import PolysynapticCircuit
from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Weights
from memristive_spinal_cord.layer2.models import ConnectionTypes


class Layer2:
    def __init__(self):
        polysynaptic_circuit = PolysynapticCircuit()
        mediator = Nucleus('Mediator')
        interneuronal_pool = Nucleus('InterneuronalPool')

        mediator.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )

        interneuronal_pool.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )

        mediator.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=polysynaptic_circuit.get_input().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )

        for tier, weight in zip(polysynaptic_circuit.get_output(), Weights.LIP.value.reverse()):
            tier.nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=interneuronal_pool.nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=weight,
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )

        mediator.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        interneuronal_pool.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()

