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
