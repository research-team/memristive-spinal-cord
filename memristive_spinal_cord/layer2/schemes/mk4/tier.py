from neucogar.Nucleus import Nucleus

from memristive_spinal_cord.layer2.models import Neurotransmitters, ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk4.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk4.components.parameters import Constants, Weights
from memristive_spinal_cord.layer2.schemes.mk4.components.synapses import Synapses


class Tier:

    # total number of neurones in the all tiers
    __number_of_neurons = 0

    @classmethod
    def add_neurons(cls, number:int):
        cls.__number_of_neurons += number

    @classmethod
    def get_number_of_neurones(cls):
        return cls.__number_of_neurons

    def __init__(self, index:int):
        """

        Args:
            index: the tier's index in order
        """
        self.__index = index
        self.__E = []
        self.__I = []
        self.__excitatory_groups = 6
        self.__inhibitory_groups = 2

        for i in range(self.__excitatory_groups):
            self.__E.append(Nucleus(nucleus_name="Tier{}E{}".format(self.__index, i)))
            self.__E[i].addSubNucleus(
                neurotransmitter=Neurotransmitters.GLU.value,
                number=Constants.NEURONS_IN_GROUP.value,
                params=Neurons.NEUCOGAR.value
            )
        for i in range(self.__inhibitory_groups):
            self.__I.append(Nucleus(nucleus_name="Tier{}I{}".format(self.__index, i)))
            self.__I[i].addSubNucleus(
                neurotransmitter=Neurotransmitters.GABA.value,
                number=Constants.NEURONS_IN_GROUP.value,
                params=Neurons.NEUCOGAR.value
            )

        self.add_neurons(Constants.NEURONS_IN_GROUP.value * (
            self.__excitatory_groups + self.__inhibitory_groups
        ))

    def get_e(self, index:int):
        """

        Args:
            index:

        Returns:
            Nucleus
        """
        return self.__E[index]

    def get_i(self, index:int):
        """

        Args:
            index:

        Returns:
            Nucleus
        """
        return self.__I[index]

    def connect_multimeters(self):
        for i in range(self.__excitatory_groups):
            self.get_e(i).nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        for i in range(self.__inhibitory_groups):
            self.get_i(i).nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()

    def set_connections(self):
        for i in [j for j in range(self.__excitatory_groups - 1) if j != 3]:
            self.get_e(i).nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.get_e(i + 1).nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EE.value[self.__index - 1][i if i != 4 else 5],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )
        for i in [3, 5]:
            self.get_e(i).nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.get_e(i - 1).nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EE.value[self.__index - 1][i if i == 3 else 6],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

        self.get_e(0).nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_e(4).nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EE.value[self.__index - 1][4],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        for i, j in zip([4, 3], [0, 1]):
            self.get_e(i).nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.get_i(j).nuclei(Neurotransmitters.GABA.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EI.value[self.__index-1][j],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

        for i, j in zip([1, 2], [0, 1]):
            self.get_i(j).nuclei(Neurotransmitters.GABA.value).connect(
                nucleus=self.get_e(i).nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=-Weights.IE.value[self.__index-1][j],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

    def connect(self, lower_tier):
        lower_tier.get_e(0).nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_e(0).nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[self.__index-2][0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.get_e(3).nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=lower_tier.get_e(3).nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[self.__index-2][1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        lower_tier.get_e(4).nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_e(0).nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[self.__index-2][2],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
