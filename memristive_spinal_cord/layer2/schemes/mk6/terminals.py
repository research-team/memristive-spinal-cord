from neucogar.Nucleus import Nucleus

from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk6.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk6.components.parameters import Weights
from memristive_spinal_cord.layer2.schemes.mk6.components.synapses import Synapses


class Terminals:

    number_of_neurons = 0

    @classmethod
    def add_number_of_neurons(cls, number: int):
        cls.number_of_neurons += number

    @classmethod
    def get_number_of_neurons(cls):
        return cls.number_of_neurons

    def __init__(self):
        self.mediator = Nucleus('Mediator')
        self.mediator.addSubNucleus(
            neurotransmitter='Glu',
            params=Neurons.NEUCOGAR.value,
            number=1
        )
        self.pool = Nucleus('InterneuronalPool')
        self.pool.addSubNucleus(
            neurotransmitter='Glu',
            params=Neurons.NEUCOGAR.value,
            number=1
        )

    def connect(self, input, output):
        for nucleus in input:
            self.mediator.nuclei('Glu').connect(
                nucleus=nucleus.nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.MR.value,
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )
        for i in range(len(output)):
            output[i].nuclei('Glu').connect(
                nucleus=self.pool.nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.P.value[i],
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )

    def connect_multimeters(self):
        for nucleus in [self.pool, self.mediator]:
            nucleus.nuclei('Glu').ConnectMultimeter()