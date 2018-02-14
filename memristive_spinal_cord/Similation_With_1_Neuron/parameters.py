from enum import Enum


class Weights(Enum):

    EE = [
        [95., 100., 100., 100., 100., 100.], # 95 100 100 100 100 100
        [100., 100., 100., 100., 100., 100.],#100
        [100., 100., 100., 100., 100., 100.],
        [100., 100., 100., 100., 100., 100.],
        [195., 100., 100., 100., 100., 100.],
        [170., 100., 100., 100., 100., 100.]
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
        [100., 100.]
    ]

    TT = [
        [60., 35., 100.],
        [60., 35., 66.2], # 60., 35., 66.2
        [52.35, 45., 100.],
        [60., 35., 65.],
        [160., 85., 10.]  #100 35 40

    ]

    # To the pool
    P = [60., 58., 40., 25., 30., 10.]

    # Mediator to PC weight
    MR = 100.

    # Spike generator weight
    SG = 200.

    INH = 100


class Constants(Enum):
    NEURONS_IN_GROUP = 1

    REFRACTORY_PERIOD = [1.8, 2.2]
    ACTION_TIME_EX = 0.5
    ACTION_TIME_INH = 5.

    GENERATOR_FREQUENCY = 40.
    SIMULATION_TIME = 250.  # milliseconds
    SPIKE_GENERATOR_TIMES = [25 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]
    SPIKE_GENERATOR_WEIGHTS = [Weights.SG.value for _ in SPIKE_GENERATOR_TIMES]

    SYNAPTIC_DELAY_EX = 0.5
    SYNAPTIC_DELAY_INH = 0.5

    RESOLUTION = 0.1
    LOCAL_NUM_THREADS = 2


class Paths(Enum):
    DATA_DIR_NAME = 'raw_data'
    FIGURES_DIR_NAME = 'images'
