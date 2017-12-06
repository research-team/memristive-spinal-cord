class NeuronGroupParams:
    def __init__(self, type, number, model):
        self._type = type
        self._number = number
        self._model = model

    def get_type(self):
        """
        Returns:
            String
        """
        return self._type

    def get_number(self):
        """
        Returns:
            Int
        """
        return self._number

    def get_model(self):
        """
        Returns:
            Dictionary
        """
        return self._model
