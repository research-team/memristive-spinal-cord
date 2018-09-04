import nest


class DummySensoryReceptor:

    def __init__(self, inversion: bool=False, time: float=200, period: float=100.,
                 stand_coef: float=0.7, rate: float=60):
        spike_times = []
        standing_time = period * stand_coef
        walking_time = period * (1 - stand_coef)
        if inversion:
            i = 0
            periods = [standing_time, walking_time]
        else:
            i = 1
            periods = [walking_time, standing_time]
        timepoint = 0.1
        while timepoint < time:
            if i:
                timepoint += periods[i]
                i = (i + 1) % 2
            else:
                spikes_at_period = int(periods[i] / 20 * rate)
                time_between_spikes = round(periods[i] / spikes_at_period, 1)
                spike_times.extend([timepoint + time_between_spikes * i for i in range(spikes_at_period)])
                timepoint += periods[i]
                i = (i + 1) % 2

        self.receptor_id = nest.Create(
            model='spike_generator',
            n=1,
            params={'spike_times': spike_times, 'spike_weights': [100. for _ in spike_times]}
        )
