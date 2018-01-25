from enum import Enum


class Weights(Enum):

    EE = [
        [95., 100., 100., 100., 100., 100.],
        [100., 100., 100., 100., 100., 100.],
        [100., 100., 100., 100., 100., 100.],
        [100., 100., 100., 100., 100., 100.],
        [100., 100., 100., 100., 100., 100.],
        [100., 100., 100., 100., 100., 100.]
    ]

    EI = [
        [70., 70.],
        [70., 70.],
        [70., 70.],
        [100., 100.],
        [100., 100.],
        [100., 100.]
    ]

    IE = [
        [100., 100.],
        [100., 100.],
        [100., 100.],
        [100., 100.],
        [100., 100.],
        [100., 100.],
    ]

    TT = [
        [60., 35., 100.],
        [60., 35., 100.],
        [60., 35., 100.],
        [60., 35., 0.],
        [60., 35., 0.]
    ]

    # To the pool
    P = [1., 1., 1., 1., 1., 1.]

    # Mediator to PC weight
    MR = 100.

    # Spike generator weight
    SG = 200.

    INH = 100


class Constants(Enum):
    NEURONS_IN_GROUP = 20

    REFRACTORY_PERIOD = [1.8, 2.2]
    ACTION_TIME_EX = 0.5
    ACTION_TIME_INH = 5.

    GENERATOR_FREQUENCY = 40.
    SIMULATION_TIME = 240.  # milliseconds
    SPIKE_GENERATOR_TIMES = [30 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]
    SPIKE_GENERATOR_WEIGHTS = [Weights.SG.value for _ in SPIKE_GENERATOR_TIMES]

    SYNAPTIC_DELAY_EX = 0.5
    SYNAPTIC_DELAY_INH = 0.5

    RESOLUTION = 0.1
    LOCAL_NUM_THREADS = 2


class Paths(Enum):
    DATA_DIR_NAME = 'raw_data'
    FIGURES_DIR_NAME = 'images'
