from enum import Enum
import pkg_resources
from memristive_spinal_cord.layer1.params.entity_params import EntityParams
from memristive_spinal_cord.frequency_generators.list import FrequencyList


class Types(Enum):
    ONE_A = 'Ia'
    TWO = 'II'


class Muscles(Enum):
    FLEX = "TA"
    EXTENS = "GM"


class Interval(Enum):
    TWENTY = 20


class Speed(Enum):
    FIFTEEN = 15
    DEFAULT = ''


class AfferentsParams(EntityParams):
    def __init__(self, speed, interval, number, type, muscle):
        self.speed = speed
        self.interval = interval
        if number <= 0 or number > 60:
            raise ValueError("Afferents.number must be greater than 0 and less or equal than 60")
        self.number = number
        self.type = type
        self.muscle = muscle

    def _create_generator_params(self, spike_times_list):
        spike_times_list = [{"spike_times": spike_times} for spike_times in spike_times_list]
        return dict(model="spike_generator", n=self.number, params=spike_times_list)


class AfferentsParamsFile(AfferentsParams):
    def __init__(self, filepath, speed, interval, number, type, muscle):
        super().__init__(speed, interval, number, type, muscle)
        self.filepath = filepath

    def _get_data_filename(self):
        # example: Ia_GM_speed15_int20.txt
        return self.type.value + "_" + self.muscle.value + "_speed" \
               + str(self.speed.value) + "_int" + str(self.interval.value) + ".txt"

    def _get_data_file(self):
        return pkg_resources.resource_filename(
            "memristive_spinal_cord",
            self.filepath + self._get_data_filename()
        )

    def to_nest_params(self):
        with open(self._get_data_file(), "r") as data_file:
            frequencies_list = []
            for line in data_file:
                frequency_list = [float(frequency) for frequency in line.strip().split()]
                frequencies_list.append(FrequencyList(self.interval.value, frequency_list))
                if len(frequencies_list) >= self.number:
                    break
        spike_times_list = [frequency_list.generate_spikes() for frequency_list in frequencies_list]
        return self._create_generator_params(spike_times_list)


def test():

    def spikes_list_from_file(filepath):
        data_filepath = pkg_resources.resource_filename('memristive_spinal_cord', filepath)
        number = 1
        with open(data_filepath, "r") as data_file:
            frequencies_list = []
            for line in data_file:
                frequency_list = [float(frequency) for frequency in line.strip().split()]
                frequencies_list.append(FrequencyList(20, frequency_list))
                if len(frequencies_list) >= number:
                    break

        return [frequency_list.generate_spikes() for frequency_list in frequencies_list]


    gm_spikes_list = spikes_list_from_file('/layer1/moraud/afferents_data/Ia_GM_speed_int20.txt')
    ta_spikes_list = spikes_list_from_file('/layer1/moraud/afferents_data/Ia_TA_speed_int20.txt')

    # print("@@@", len(gm_spikes_list))
    # print("###", spikes_list[0])
    import pylab
    pylab.figure()
    gm_spikes = gm_spikes_list[0]
    ta_spikes = ta_spikes_list[0]
    pylab.plot(gm_spikes, [1] * len(gm_spikes), ".")
    pylab.plot(ta_spikes, [1.1] * len(ta_spikes), ".")
    pylab.show()


test()
