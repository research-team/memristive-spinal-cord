import nest


class NeuronNetwork:
    def __init__(self, entity_params, connection_params):
        self._entitites = dict()
        self._entities_params = entity_params
        self._populate()
        self._connectome(connection_params)

    def _populate(self):
        for entity_name, entity_params in self._entities_params.items():
            self.create_entity(entity_name, entity_params)

    def create_entity(self, entity_name, entity_params):
        self._entitites[entity_name] = nest.Create(**entity_params)

    def get_entity(self, entity_name):
        """
        Args:
            entity_name (Layer1Entities)
        Return:
            Nest
        """
        return self._entitites[entity_name]

    def connect_by_params(self, connection_params):
        """
        Args:
            connection_params (ConnectionParams)
        """
        nest_params = dict(connection_params)
        nest_params['pre'] = self.get_entity(nest_params['pre'])
        nest_params['post'] = self.get_entity(nest_params['post'])
        nest.Connect(**nest_params)

    def _connectome(self, params_storage):
        """
        Args:
            params_storage (ConnectionParamsStorage)
        """
        for connection_params in params_storage.items():
            self.connect_by_params(connection_params)
