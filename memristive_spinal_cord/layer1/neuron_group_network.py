from neucogar.Nucleus import Nucleus


class NeuronGroupNetwork:
    def __init__(self):
        self._neuron_number = 0

    def create_neuron_group(self, group_name, group_params):
        """
        Function for building spike diagrams

        Args:
            group_name (str)
            group_params (NeuronGroupParameters)
        """

        neuron_group = Nucleus(group_name)
        neuron_group.addSubNucleus(group_params.get_type(),
                                   params=group_params.get_model(),
                                   number=group_params.get_number())
        self._neuron_number += group_params.get_number()
        return neuron_group

    def get_neuron_number(self):
        return self._neuron_number
