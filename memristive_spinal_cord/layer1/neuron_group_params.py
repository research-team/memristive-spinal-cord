class NeuronGroupParams:
    def __init__(self, display_name, type, number=None, model=None):
        self._display_name = display_name
        self._type = type
        self._number = number
        self._model = model

    def get_display_name(self):
        """
        Returns:
            String
        """
        return self._display_name

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

    def set_number(self, number):
        """
        Args:
            number (Int)
        """
        self._number = number

    def get_model(self):
        """
        Returns:
            Dictionary
        """
        return self._model

    def set_model(self, model):
        """
        Args:
            model (dict)
        """
        self._model = model
