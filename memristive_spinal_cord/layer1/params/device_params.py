from memristive_spinal_cord.layer1.params.entity_params import EntityParams


class DeviceParams(EntityParams):
    def __init__(self, name, storage_dir, params=None, number=1):
        self.name = name
        self.storage_dir = storage_dir
        self.params = params
        self.number = number

    def to_nest_params(self):
        model_params = dict(label=self.storage_dir + "/" + self.name)  # Label is the name of the file
        model_params.update(self.get_default_params())
        if self.params is not None:
            model_params.update(self.params)
        model_params.update()
        return dict(n=self.number, model=self.get_model(), params=model_params)

    def get_default_params(self):
        raise NotImplementedError("Using of abstract method of " + self.__name__)

    def get_model(self):
        raise NotImplementedError("Using of abstract method of " + self.__name__)


class MultimeterParams(DeviceParams):
    def get_default_params(self):
        return {
            'withgid': True,  # Write neuron global ID to the file
            'withtime': True,  # Write time to the file
            'to_file': True,  # Flag - is writing to file
            'to_memory': False,  # Flag - is writing to RAM
            'interval': 0.1,  # Interval (ms) of getting data from neurons
            'record_from': ['V_m']  # Recording values (V_m is membrane potential)
        }

    def get_model(self):
        return 'multimeter'


class DetectorParams(DeviceParams):
    def get_default_params(self):
        return {
            'withgid': True,  # Write neuron global ID to the file
            'withtime': True,  # Write time to the file
            'to_file': True,  # Flag - is writing to file
            'to_memory': False,  # Flag - is writing to RAM
        }

    def get_model(self):
        return 'spike_detector'
