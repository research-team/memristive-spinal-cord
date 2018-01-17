from enum import Enum


class Weights(Enum):

    EE = [
        [100., 100., 100., 95., 100., 100., 100.],
        [100., 100., 100., 95., 100., 100., 100.],
        [100., 100., 100., 95., 100., 100., 100.],
        [100., 100., 100., 95., 100., 100., 100.],
        [100., 100., 100., 95., 100., 100., 100.],
        [93., 93.]
    ]

    EI = [
        [100., 100.],
        [100., 100.],
        [100., 100.],
        [100., 100.],
        [100., 100.],
        100.
    ]

    IE = [
        [40., 45.],
        [40., 45.],
        [40., 45.],
        [40., 45.],
        [40., 45.],
        0.
    ]

    TT = [
        [50., 100., 40.],
        [50., 100., 40.],
        [50., 100., 40.],
        [50., 100., 40.],
        [45., 100., 35.],
        100.
    ]

    # To the pool
    P = [13., 13., 13., 13., 13., 13.]

    # Mediator to PC weight
    MR = 100.

    # Spike generator weight
    SG = 200.


class Constants(Enum):
    NEURONS_IN_GROUP = 1

    REFRACTORY_PERIOD = [1.8, 2.2]
    ACTION_TIME_EX = 0.5
    ACTION_TIME_INH = 5.

    GENERATOR_FREQUENCY = 40.
    SIMULATION_TIME = 450.  # milliseconds
    SPIKE_GENERATOR_TIMES = [75 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]
    SPIKE_GENERATOR_WEIGHTS = [Weights.SG.value for _ in SPIKE_GENERATOR_TIMES]

    SYNAPTIC_DELAY_EX = 0.85
    SYNAPTIC_DELAY_INH = 0.85

    RESOLUTION = 0.1
    LOCAL_NUM_THREADS = 2


class Paths(Enum):
    DATA_DIR_NAME = 'raw_data'
    FIGURES_DIR_NAME = 'images'
