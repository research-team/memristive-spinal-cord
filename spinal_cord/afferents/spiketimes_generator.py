from spinal_cord.namespace import Afferent, Muscle, Interval, Speed
from pkg_resources import resource_filename
import os


class AfferentSpikeTimeGenerator:

    def generate_spikes(frequency_list: list, interval: Interval) -> list:
        """
        Generates a list of spikes by using its own frequency list.

        Returns:
            list: the list of spike times

        """
        spike_times = []
        # initial time
        time = 0.0
        charge = 0.0

        interval = interval.value
        for frequency in frequency_list:
            spikes_at_interval = int(interval / 1000 * frequency)

            charge += interval / 1000 * frequency - spikes_at_interval
            if charge > 1:
                charge -= 1
                spikes_at_interval += 1

            if spikes_at_interval > 0:
                time_between_spikes = interval / spikes_at_interval
                time -= time_between_spikes / 2  # shifting time to place spikes closer to the center
                spike_times.extend(
                    [round(time + time_between_spikes * (n + 1), 2) for n in range(spikes_at_interval)])
                time += time_between_spikes / 2  # shifting back
            time += interval
        return spike_times

    @staticmethod
    def get_spiketimes_list(
            muscle: Muscle,
            afferent: Afferent,
            number: int,
            speed: Speed,
            interval: Interval,
            datapath: str
    ) -> list:
        """
        Reads the experimental data from a file and returns a list of spike times for a specific afferent
        """
        if number in range(0, 61):
            # example: Ia_GM_speed15_int20.txt
            filename = '{afferent}_{muscle}_speed{speed}_int{interval}.txt'.format(
                afferent=afferent.value, muscle=muscle.value, speed=speed.value, interval=interval.value)
            filepath = resource_filename('afferents', os.path.join(datapath, filename))

            print('Getting data from {}'.format(filename))
            with open(filepath, 'r') as data_file:
                frequencies_list = [[float(value) for value in line.strip().split()] for line in data_file.readlines()]
            spike_times_list = [AfferentSpikeTimeGenerator.generate_spikes(frequencies, interval) for frequencies in frequencies_list][:number]
            return [{'spike_times': spike_times} for spike_times in spike_times_list]
        else:
            raise ValueError("Wrong afferents number")


def test() -> None:
    spike_generator = AfferentSpikeTimeGenerator()
    flex_spikes_list = spike_generator.get_spiketimes_list(
        muscle=Muscle.FLEX,
        number=1,
        afferent=Afferent.IA,
    )
    print(flex_spikes_list)
