from neucogar.Nucleus import Nucleus
from neucogar.api_kernel import CreateNetwork
from neucogar import api_kernel

from memristive_spinal_cord.layer2.models import Neurotransmitters, ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk4.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk4.components.parameters import Weights, Constants
from memristive_spinal_cord.layer2.schemes.mk4.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.mk4.pc import PolysynapticCircuit


class Layer2:
    def __init__(self):
        self.__polysynaptic_circuit = PolysynapticCircuit()
        self.__mediator = Nucleus("Mediator")
        self.__interneuronal_pool = Nucleus("InterneuronalPool")

        self.__mediator.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )
        self.__interneuronal_pool.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )

        CreateNetwork(simulation_neuron_number=PolysynapticCircuit.get_number_of_neurons() + 2)
        self.set_connections()
        self.connect_multimeters()
        self.connect_spike_generator()
        # self.connect_noise_generator()

    def set_connections(self):
        self.__mediator.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.__polysynaptic_circuit.get_input().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )
        for tier, weight in zip(self.__polysynaptic_circuit.get_output(), Weights.P.value):
            tier.nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.__interneuronal_pool.nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=weight,
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )
        for tier in self.__polysynaptic_circuit.get_tiers():
            tier.set_connections()
        self.__polysynaptic_circuit.set_connections()

    def connect_multimeters(self):
        self.__mediator.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.__interneuronal_pool.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        for tier in self.__polysynaptic_circuit.get_tiers():
            tier.connect_multimeters()
        self.__polysynaptic_circuit.connect_multimeters()

    def connect_spike_generator(self):
        spike_generator = api_kernel.NEST.Create('spike_generator', 1, {
            'spike_times': Constants.SPIKE_GENERATOR_TIMES.value,
            'spike_weights': Constants.SPIKE_GENERATOR_WEIGHTS.value
        })

        api_kernel.NEST.Connect(
            spike_generator,
            self.__mediator.nuclei(Neurotransmitters.GLU.value).getNeurons())

    def connect_noise_generator(self):
        self.__mediator.nuclei(Neurotransmitters.GLU.value).ConnectPoissonGenerator(
            weight=Weights.SG.value,
            start=1,
            stop=5,
            rate=100
        )