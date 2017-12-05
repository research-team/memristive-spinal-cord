class NeuronGroupParams:
    def __init__(self, type, model, number):
        self._type = type
        self._model = model
        self._number = number

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
