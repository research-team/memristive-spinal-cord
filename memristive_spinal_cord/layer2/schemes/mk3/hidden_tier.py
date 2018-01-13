from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.mk3.components.parameters import Constants
from memristive_spinal_cord.layer2.schemes.mk3.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk3.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.mk3.components.parameters import HiddenWeights
from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk3.tier import Tier


class HiddenTier:
    __number_of_neurons = 0

    @classmethod
    def add_neurons(cls, number: int):
        cls.__number_of_neurons += number

    @classmethod
    def get_number_of_neurons(cls):
        return cls.__number_of_neurons

    def __init__(self, index:int):

        self.index = index

        self.left_excitatory = Nucleus(nucleus_name='HiddenTier{}LeftExcitatory'.format(self.index))
        self.right_excitatory = Nucleus(nucleus_name='HiddenTier{}RightExcitatory'.format(self.index))
        self.right_inhibitory = Nucleus(nucleus_name='HiddenTier{}RightInhibitory'.format(self.index))
        self.left_inhibitory = Nucleus(nucleus_name='HiddenTier{}LeftInhibitory'.format(self.index))

        self.left_excitatory.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )
        self.right_excitatory.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )
        self.left_inhibitory.addSubNucleus(
            neurotransmitter=Neurotransmitters.GABA.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )
        self.right_inhibitory.addSubNucleus(
            neurotransmitter=Neurotransmitters.GABA.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.add_neurons(Constants.NEURONS_IN_GROUP.value * 4)

    def connect_multimeters(self):
        self.left_excitatory.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.right_excitatory.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.left_inhibitory.nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()
        self.right_inhibitory.nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()

    def set_connections(self):
        self.left_excitatory.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_excitatory.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.LR.value[::-1][self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.right_excitatory.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.left_excitatory.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.RL.value[::-1][self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.left_excitatory.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_inhibitory.nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GABAERGIC.value,
            weight=HiddenWeights.RIRE.value[::-1][self.index - 1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.right_inhibitory.nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=self.right_excitatory.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=-HiddenWeights.RIRE.value[::-1][self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def get_index(self): return self.index

    def get_left_excitatory(self): return self.left_excitatory

    def get_left_inhibitory(self): return self.left_inhibitory

    def connect(self, upper_tier:Tier, lower_tier:Tier):
        self.get_left_inhibitory().nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=lower_tier.get_right_group().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=-HiddenWeights.IRD.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        upper_tier.get_control_group().nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_left_inhibitory().nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.CID.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        self.get_left_excitatory().nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=upper_tier.get_control_group().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.LCU.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        lower_tier.get_control_group().nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_left_excitatory().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.CDL.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
