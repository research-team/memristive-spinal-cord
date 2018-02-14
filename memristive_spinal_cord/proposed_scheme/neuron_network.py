import nest


class NeuronNetwork:
    def __init__(self, entity_params: dict, connection_params_list: list) -> None:
        """
        Args:
            entity_params: united params for devices, afferents and neuron groups
            connection_params_list: a list of connections between neuron groups with other neuron groups and devices
        """
        self._entitites = dict()
        self._entities_params = entity_params
        self._populate()
        self._connectome(connection_params_list)

    def _populate(self) -> None:
        """
        Creates entities
        Returns:
            None
        """
        for entity_name, entity_params in self._entities_params.items():
            self.create_entity(entity_name, entity_params)

    def create_entity(self, entity_name: str, entity_params: dict) -> None:
        """
        Args:
            entity_name (str): name of the specific item (i.e. "flex-inter1A-multimeter")
            entity_params (dict): a dictionary with parameters
        Returns:
            None
        """
        self._entitites[entity_name] = nest.Create(**entity_params)

    def get_entity(self, entity_name: str) -> list:
        """
        Args:
            entity_name (Layer1Entities)
        Return:
            list: a list of global IDs of created nodes
        """
        return self._entitites[entity_name]

    def connect_by_params(self, connection_params: dict) -> None:
        """
        Args:
            connection_params (dict)
        """
        print(connection_params)
        nest_params = dict(connection_params)
        nest_params['pre'] = self.get_entity(nest_params['pre'])
        nest_params['post'] = self.get_entity(nest_params['post'])
        nest.Connect(**nest_params)

    def _connectome(self, connection_params_list) -> None:
        """
        Args:
            connection_params_list (list)
        """
        for connection_params in connection_params_list:
            self.connect_by_params(connection_params)
