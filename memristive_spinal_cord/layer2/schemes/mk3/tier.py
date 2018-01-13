from neucogar.Nucleus import Nucleus

from memristive_spinal_cord.layer2.models import Neurotransmitters, ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk3.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk3.components.parameters import Constants, TierWeights
from memristive_spinal_cord.layer2.schemes.mk3.components.synapses import Synapses


class Tier:

    def __init__(self, index):

        self.index = index

        self.left_group = Nucleus(nucleus_name='Tier{}Left'.format(str(self.index)))
        self.right_group = Nucleus(nucleus_name='Tier{}Right'.format(str(self.index)))
        self.inhibitory_group = Nucleus(nucleus_name='Tier{}Inhibitory'.format(str(self.index)))
        self.control_group = Nucleus(nucleus_name='Tier{}Control'.format(str(self.index)))

        self.left_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.right_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.inhibitory_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GABA.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.control_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

    def connect_multimeters(self):
        self.right_group.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.left_group.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.inhibitory_group.nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()
        self.control_group.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()

    def set_connections(self):
        self.right_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.left_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=TierWeights.RL.value[0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        self.left_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=TierWeights.LR.value[0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        self.right_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.inhibitory_group.nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=TierWeights.RI.value[0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        self.inhibitory_group.nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=self.left_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=-TierWeights.IL.value[0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        self.control_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=TierWeights.CR.value[0],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def get_control_group(self): return self.control_group

    def get_left_group(self): return self.left_group

    def get_right_group(self): return self.right_group
