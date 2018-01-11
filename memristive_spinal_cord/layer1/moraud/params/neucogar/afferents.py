import memristive_spinal_cord.layer1.afferents as afferents
from memristive_spinal_cord.layer1.moraud.afferents import Layer1Afferents

afferent_params = dict()

afferent_filepath = "/layer1/moraud/afferents_data/"

generator_number_1a = 20
afferent_params[Layer1Afferents.FLEX_1A] = dict(
    model="spike_generator",
    n=generator_number_1a,
    params=afferents.AfferentsFile.get_nest_spike_times(
        filepath=afferent_filepath,
        speed=afferents.Speed.FIFTEEN,
        interval=afferents.Interval.TWENTY,
        type=afferents.Types.ONE_A,
        muscle=afferents.Muscles.FLEX,
        number=generator_number_1a,
    ),
)
afferent_params[Layer1Afferents.EXTENS_1A] = dict(
    model="spike_generator",
    n=generator_number_1a,
    params=afferents.AfferentsFile.get_nest_spike_times(
        filepath=afferent_filepath,
        speed=afferents.Speed.FIFTEEN,
        interval=afferents.Interval.TWENTY,
        type=afferents.Types.ONE_A,
        muscle=afferents.Muscles.EXTENS,
        number=generator_number_1a,
    ),
)

generator_number_2 = 20
afferent_params[Layer1Afferents.FLEX_2] = dict(
    model="spike_generator",
    n=generator_number_2,
    params=afferents.AfferentsFile.get_nest_spike_times(
        filepath=afferent_filepath,
        speed=afferents.Speed.FIFTEEN,
        interval=afferents.Interval.TWENTY,
        type=afferents.Types.TWO,
        muscle=afferents.Muscles.FLEX,
        number=generator_number_2,
    ),
)
afferent_params[Layer1Afferents.EXTENS_2] = dict(
    model="spike_generator",
    n=generator_number_2,
    params=afferents.AfferentsFile.get_nest_spike_times(
        filepath=afferent_filepath,
        speed=afferents.Speed.FIFTEEN,
        interval=afferents.Interval.TWENTY,
        type=afferents.Types.TWO,
        muscle=afferents.Muscles.EXTENS,
        number=generator_number_2,
    ),
)
