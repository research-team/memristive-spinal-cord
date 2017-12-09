from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer1.rybak.params.neucogar.neuron_groups import layer1_neuron_group_params


class NeuronNetwork:
    def __init__(self):
        self._groups = dict()

    def create_neuron_group(self, group_name, group_params):
        """
        Args:
            group_name (Layer1NeuronGroupNames)
            group_params (NeuronGroupParameters)
        """

        neuron_group = Nucleus(group_name.value)
        neuron_group.addSubNucleus(group_params.get_type(),
                                   params=group_params.get_model(),
                                   number=group_params.get_number())
        self._groups[group_name] = neuron_group
        return neuron_group

    def get_neuron_number(self):
        """
        Return:
            Int
        """
        neuron_number = 0
        for group_name, group in self._groups.items():
            neuron_number += self.get_neuron_group_nuclei(group_name).getNeuronNumber()
        return neuron_number

    def get_neuron_group(self, group_name):
        """
        Args:
            group_name (Layer1NeuronGroupNames)
        Return:
            Nucleus
        """
        return self._groups[group_name]

    def get_neuron_group_nuclei(self, group_name):
        """
        Args:
            group_name (Layer1NeuronGroupNames)
        Return:
            Nucleus
        """
        group = self.get_neuron_group(group_name)
        return group.nuclei(layer1_neuron_group_params[group_name].get_type())

    def connect(self, source, target, synapse, weight):
        """
        Args:
            source (Layer1NeuronGroupNames)
            target (Layer1NeuronGroupNames)
            synapse (SynapseModel)
            weight (double)
        """
        source_nuclei = self.get_neuron_group_nuclei(source)
        target_nuclei = self.get_neuron_group_nuclei(target)
        source_nuclei.connect(nucleus=target_nuclei, synapse=synapse, weight=weight)
