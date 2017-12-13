from enum import Enum
import pkg_resources
from memristive_spinal_cord.frequency_generators.list import FrequencyList


class Types(Enum):
    ONE_A = 'Ia'
    TWO = 'II'


class Muscles(Enum):
    # flexor
    L = "TA"
    # extensor
    R = "GM"


class Interval(Enum):
    TWENTY = 20


class Speed(Enum):
    FIFTEEN = 15


class Afferents:
    def __init__(self, speed, interval, number):
        self._speed = speed
        self._interval = interval
        if number <= 0 or number > 60:
            raise ValueError("Afferents.number must be greater than 0 and less or equal than 60")
        self._number = number

    def create_generator_params(self, type, muscle):
        raise NotImplementedError("Using of abstract method of " + self.__name__)

    def _create_generator_params(self, spike_times_list):
        spike_times_list = [{"spike_times": spike_times} for spike_times in spike_times_list]
        return dict(model="spike_generator", number=self._number, params=spike_times_list)


class AfferentsFile(Afferents):
    def _get_data_filename(self, type, muscle):
        # example: fr_Ia_GM_speed15_interval20.txt
        return type.value + "_" + muscle.value + "_speed" + str(self._speed.value) + "_int" + str(
            self._interval.value) + ".txt"

    def _get_data(self, type, muscle):
        return pkg_resources.resource_filename(
            "memristive_spinal_cord",
            "/layer1/moraud/afferents/data/" + self._get_data_filename(type, muscle)
        )

    def create_generator_params(self, type, muscle):
        with open(self._get_data(type, muscle), "r") as data_file:
            frequencies_list = []
            for line in data_file:
                frequency_list = [float(frequency) for frequency in line.strip().split()]
                frequencies_list.append(FrequencyList(self._interval.value, frequency_list))
                if len(frequencies_list) >= self._number:
                    break
        spike_times_list = [frequency_list.generate_spikes() for frequency_list in frequencies_list]
        return self._create_generator_params(spike_times_list)


class Test:
    def __init__(self) -> None:
        super().__init__()
        data_filepath = pkg_resources.resource_filename(
            "memristive_spinal_cord",
            "/layer1/moraud/afferents/data/Ia_GM_speed15_int20.txt"
        )
        number = 3
        with open(data_filepath, "r") as data_file:
            frequencies_list = []
            for line in data_file:
                frequency_list = [float(frequency) for frequency in line.strip().split()]
                frequencies_list.append(FrequencyList(20, frequency_list))
                if len(frequencies_list) >= number:
                    break

        spikes_list = [frequency_list.generate_spikes() for frequency_list in frequencies_list]
        print("@@@", len(spikes_list))
        print("###", spikes_list[0])

# Test()
