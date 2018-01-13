from enum import Enum
import pkg_resources
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


class AfferentsFile:
    @staticmethod
    def get_nest_spike_times(filepath, speed, interval, number, type, muscle):
        if number <= 0 or number > 60:
            raise ValueError("AfferentsFile.number must be greater than 0 and less or equal than 60")

        with open(AfferentsFile._get_data_file(filepath, type, muscle, speed, interval), "r") as data_file:
            frequencies_list = []
            for line in data_file:
                frequency_list = [float(frequency) for frequency in line.strip().split()]
                frequencies_list.append(FrequencyList(interval.value, frequency_list))
                if len(frequencies_list) >= number:
                    break
        spike_times_list = [frequency_list.generate_spikes() for frequency_list in frequencies_list]
        return [{"spike_times": spike_times} for spike_times in spike_times_list]

    @staticmethod
    def _get_data_file(filepath, type, muscle, speed, interval):
        # example: Ia_GM_speed15_int20.txt
        filename = type.value + "_" + muscle.value + "_speed" \
                   + str(speed.value) + "_int" + str(interval.value) + ".txt"
        return pkg_resources.resource_filename(
            "memristive_spinal_cord",
            filepath + filename
        )


def test():
    flex_spikes_list = AfferentsFile.get_nest_spike_times(
        '/layer1/moraud/afferents_data/',
        Speed.DEFAULT,
        Interval.TWENTY,
        1,
        Types.ONE_A,
        Muscles.FLEX
    )

    extens_spikes_list = AfferentsFile.get_nest_spike_times(
        '/layer1/moraud/afferents_data/',
        Speed.DEFAULT,
        Interval.TWENTY,
        1,
        Types.ONE_A,
        Muscles.FLEX
    )

    # print("@@@", len(gm_spikes_list))
    # print("###", spikes_list[0])
    import pylab
    pylab.figure()
    flex_spikes = flex_spikes_list[0]['spike_times']
    extens_spikes = extens_spikes_list[0]['spike_times']
    pylab.plot(flex_spikes, [1] * len(flex_spikes), ".")
    pylab.plot(extens_spikes, [1.1] * len(extens_spikes), ".")
    pylab.show()

# test()
