class ConnectionParamsStorage:
    def __init__(self):
        self._storage = []

    def add(self, params):
        """
        Args:
            params (ConnectionParams)
        """
        self._storage.append(params)

    def items(self):
        """
        Return:
            (ConnectionParams[])
        """
        return self._storage
