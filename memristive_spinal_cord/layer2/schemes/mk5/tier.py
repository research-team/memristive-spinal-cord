from neucogar.Nucleus import Nucleus

from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk5.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk5.components.parameters import Constants, Weights
from memristive_spinal_cord.layer2.schemes.mk5.components.synapses import Synapses


class Tier:

    __number_of_neurons = 0
    excitatory_groups = 5
    inhibitory_groups = 1

    @classmethod
    def add_neurons(cls, number: int):
        cls.__number_of_neurons += number

    @classmethod
    def get_number_of_neurons(cls):
        return cls.__number_of_neurons

    def __init__(self, index: int):
        self.index = index
        self.e = []
        self.i = Nucleus('I0')
        for i in range(self.excitatory_groups):
            self.e.append(Nucleus('E{}'.format(i)))
            self.e[i].addSubNucleus(
                neurotransmitter='Glu',
                number=Constants.NEURONS_IN_GROUP.value,
                params=Neurons.NEUCOGAR.value
            )
        self.i.addSubNucleus(
            neurotransmitter='GABA',
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )
        self.add_neurons(Constants.NEURONS_IN_GROUP.value * (
            self.excitatory_groups + self.inhibitory_groups
        ))

    def get_e(self, index: int):
        """

        Args:
            index:

        Returns:
            Nucleus
        """
        return self.e[index]

    def get_i(self):
        """

        Returns:
            Nucleus
        """
        return self.i

    def connect_multimeters(self):
        for i in range(self.excitatory_groups):
            self.e[i].nuclei('Glu').ConnectMultimeter()
        self.i.nuclei('GABA').ConnectMultimeter()

    def set_connections(self):
        for i in [0, 1, 3]:
            self.get_e(i).nuclei('Glu').connect(
                nucleus=self.get_e(i+1).nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EE.value[self.index-1][i],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )
        for i in [2, 4]:
            self.get_e(i).nuclei('Glu').connect(
                nucleus=self.get_e(i-1).nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.EE.value[self.index-1][i],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )
        self.get_e(0).nuclei('Glu').connect(
            nucleus=self.get_e(3).nuclei('Glu'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EE.value[self.index-1][5],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.get_e(3).nuclei('Glu').connect(
            nucleus=self.get_i().nuclei('GABA'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.EI.value[self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.get_i().nuclei('GABA').connect(
            nucleus=self.get_e(1).nuclei('Glu'),
            synapse=Synapses.GABAERGIC.value,
            weight=Weights.IE.value[self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def connect(self, lower_tier):
        for i in [0, 3]:
            lower_tier.get_e(i).nuclei('Glu').connect(
                nucleus=self.get_e(0).nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.TT.value[self.index-1][0 if i == 0 else 1],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )
        self.get_e(2).nuclei('Glu').connect(
            nucleus=self.get_e(2).nuclei('Glu'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.TT.value[self.index-1][2],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )