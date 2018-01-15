from enum import Enum


class Weights(Enum):

    EE = [
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
    ]

    EI = [
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ]

    IE = [
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ]

    TT = [
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
    ]

    # To the pool
    P = [0., 0., 0., 0., 0., 0.]

    # Mediator to PC weight
    MR = 150.

    # Spike generator weight
    SG = 200.


class Constants(Enum):
    NEURONS_IN_GROUP = 20

    REFRACTORY_PERIOD = [1.8, 2.2]
    ACTION_TIME_EX = 0.5
    ACTION_TIME_INH = 5.

    SIMULATION_TIME = 200.  # milliseconds
    SPIKE_GENERATOR_TIMES = [25 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]
    SPIKE_GENERATOR_WEIGHTS = [Weights.SG.value for _ in SPIKE_GENERATOR_TIMES]

    SYNAPTIC_DELAY_EX = [1.1, 1.3]
    SYNAPTIC_DELAY_INH = [1.1, 1.2]

    RESOLUTION = 0.1
    LOCAL_NUM_THREADS = 2


class Paths(Enum):
    DATA_DIR_NAME = 'raw_data'
    FIGURES_DIR_NAME = 'images'
