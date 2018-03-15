import nest
from spinal_cord.afferents.spiketimes_generator import AfferentSpikeTimeGenerator
from spinal_cord.namespace import Afferent, Muscle, Interval, Speed


class Receptor:

    def __init__(
            self,
            muscle: Muscle,
            afferent: Afferent,
            number: int=60,
            speed: Speed=Speed.DEFAULT,
            interval: Interval=Interval.DEFAULT,
            datapath: str='data'
    ):
        self.receptor_ids = nest.Create(
            model='spike_generator',
            n=number,
            params=AfferentSpikeTimeGenerator.get_spiketimes_list(
                muscle=muscle,
                afferent=afferent,
                number=number,
                speed=speed,
                interval=interval,
                datapath=datapath
            )
        )


class DummySensoryReceptor:

    def __init__(self, muscle: Muscle, time: float=20000, period: float=1000., stand_coef: float=0.7, rate: float=60):
        self.muscle = muscle
        spike_times = []
        standing_time = period * stand_coef
        walking_time = period * (1 - stand_coef)
        periods = [standing_time, walking_time] if muscle == Muscle.EXTENS else [walking_time, standing_time]
        timepoint = 0.1
        i = 1 if muscle == Muscle.FLEX else 0
        while timepoint < time:
            if i:
                timepoint += periods[i]
                i = (i + 1) % 2
            else:
                spikes_at_period = int(periods[i] / 1000 * rate)
                time_between_spikes = round(periods[i] / spikes_at_period, 1)
                spike_times.extend([timepoint + time_between_spikes * i for i in range(spikes_at_period)])
                timepoint += periods[i]
                i = (i + 1) % 2

        self.receptor_id = nest.Create(
            model='spike_generator',
            n=1,
            params={'spike_times': spike_times, 'spike_weights': [100. for _ in spike_times]}
        )

