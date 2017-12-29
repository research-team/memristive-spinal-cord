from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Constants
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import HiddenWeights
from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.hidden_tiers.tier import Tier


class HiddenTier:

    def __init__(self, index:int):

        self.index = index

        self.left_excitatory = Nucleus(nucleus_name='Tier{}LeftExcitatory'.format(self.index))
        self.right_excitatory = Nucleus(nucleus_name='Tier{}RightExcitatory'.format(self.index))
        self.right_inhibitory = Nucleus(nucleus_name='Tier{}RightInhibitory'.format(self.index))
        self.left_inhibitory = Nucleus(nucleus_name='Tier{}LeftInhibitory'.format(self.index))

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

        self.left_excitatory.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.right_excitatory.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.left_inhibitory.nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()
        self.right_inhibitory.nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()

        self.left_excitatory.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_excitatory.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.LR.value.reverse()[self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.right_excitatory.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.left_excitatory.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.RL.value.reverse()[self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.right_excitatory.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.left_inhibitory.nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GABAERGIC.value,
            weight=HiddenWeights.RI.value.reverse()[self.index - 1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
        self.left_inhibitory.nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=self.right_excitatory.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=HiddenWeights.IR.value.reverse()[self.index-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def get_index(self): return self.index

    def get_right_excitatory(self): return self.right_excitatory

    def get_right_inhibitory(self): return self.right_inhibitory

    def connect(self, upper_tier:Tier, lower_tier:Tier):

        self.get_right_excitatory().nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=upper_tier.get_right_group().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.RRU.value.reverse()[self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        self.get_right_inhibitory().nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=lower_tier.get_right_group().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=HiddenWeights.IRD.value.reverse()[self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        upper_tier.get_right_group().nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_right_inhibitory().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.RID.value.reverse()[self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        lower_tier.get_right_group().nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.get_right_excitatory().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=HiddenWeights.RDR.value.reverse()[self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )
