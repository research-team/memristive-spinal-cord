from neucogar.Nucleus import Nucleus

from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk6.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk6.components.parameters import Constants, Weights
from memristive_spinal_cord.layer2.schemes.mk6.components.synapses import Synapses


class Tier:
    number_of_neurons = 0

    @classmethod
    def add_neurons(cls, number: int):
        cls.number_of_neurons += number

    @classmethod
    def get_number_of_neurons(cls):
        return cls.number_of_neurons

    def __init__(self, index: int):
        self.index = index
        self.excitatory = [Nucleus('Tier{}E{}'.format(index, i)) for i in range(6)]
        for nucleus in self.excitatory:
            nucleus.addSubNucleus(
                neurotransmitter='Glu',
                number=Constants.NEURONS_IN_GROUP.value,
                params=Neurons.NEUCOGAR.value
            )
        self.inhibitory = [Nucleus('Tier{}I{}'.format(index, i)) for i in range(2)]
        for nucleus in self.inhibitory:
            nucleus.addSubNucleus(
                neurotransmitter='GABA',
                number=Constants.NEURONS_IN_GROUP.value,
                params=Neurons.NEUCOGAR.value
            )
        self.add_neurons(Constants.NEURONS_IN_GROUP.value * 8)

    def set_interconnections(self):
        for i in range(6):
            self.excitatory[i].nuclei('Glu').connect(
                nucleus=self.excitatory[i+1 if i in [0, 1, 2, 4] else i-1].nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EE.value[self.index][i],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )
        self.excitatory[0].nuclei('Glu').connect(
            nucleus=self.excitatory[4].nuclei('Glu'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EE.value[self.index][6],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        for i in range(2):
            self.excitatory[4 if i == 0 else 3].nuclei('Glu').connect(
                nucleus=self.inhibitory[i].nuclei('GABA'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EI.value[self.index][i],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )
            self.inhibitory[i].nuclei('GABA').connect(
                nucleus=self.excitatory[1 if i == 0 else 3].nuclei('Glu'),
                synapse=Synapses.GABAERGIC.value,
                weight=-Weights.IE.value[self.index][i],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

    def connect_multimeters(self):
        for nucleus in self.excitatory:
            nucleus.nuclei('Glu').ConnectMultimeter()
        for nucleus in self.inhibitory:
            nucleus.nuclei('GABA').ConnectMultimeter()

    def get_e(self, index):
        """

        Args:
            index:

        Returns:
            Nucleus
        """
        return self.excitatory[index]

    def connect(self, tier):
        self.excitatory[4].nuclei('Glu').connect(
            nucleus=tier.get_e(0).nuclei('Glu'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[self.index][0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        tier.get_e(3).nuclei('Glu').connect(
            nucleus=self.excitatory[3].nuclei('Glu'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[self.index][0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )