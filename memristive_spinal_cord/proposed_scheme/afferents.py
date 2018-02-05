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
    def get_nest_spike_times(
            filepath: str, speed: Speed, interval: Interval, number: int, type: Types, muscle: Muscles) -> list:
        """
        Reads the experimental data from a file and returns a list of spike times
        Args:
            filepath: path to the directory contains the datafiles
            speed: not sure about meaning but this is a part of the filename
            interval: intercal between stimulations, also part of the filename
            number: a number of one of 60 afferents in the datafile
            type: Ia or II afferent type
            muscle: TA (Tibialis anterior?) or GM (Gluteus Maximus?)

        Returns:
            list
        """
        if number <= 0 or number > 60:
            raise ValueError("AfferentsFile.number must be greater than 0 and less or equal than 60")
        print('Get data from: {}'.format(AfferentsFile._get_data_file(filepath, type, muscle, speed, interval)))
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
    def _get_data_file(filepath: str, type: Types, muscle: Muscles, speed: Speed, interval: Interval) -> str:
        """

        Args:
            filepath: path to the directory contains the file
            type: Ia or II afferent type
            muscle: TA (Tibialis anterior?) or GM (Gluteus Maximus?)
            speed: a part of the filename
            interval: a part of the filename

        Returns:
            str: name of the file
        """
        # example: Ia_GM_speed15_int20.txt
        filename = type.value + "_" + muscle.value + "_speed" \
                   + str(speed.value) + "_int" + str(interval.value) + ".txt"
        return pkg_resources.resource_filename(
            "memristive_spinal_cord",
            filepath + filename
        )


def test() -> None:
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
        Muscles.EXTENS
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
