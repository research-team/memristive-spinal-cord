from enum import Enum


class Weights(Enum):

    EE = [
        [0, 0],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
    ]

    EI = [
        0,
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ]

    IE = [
        0,
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
    ]

    TT = [
        0.,
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ]

    # To the pool
    PE = [1., 1., 1., 1., 1., 1., 1.]
    PI = 70

    # From pool to Mn
    PM = {
        'Extensor': 0,
        'Flexor': 0
    }

    # From pool to Ia

    PIa = {
        'Extensor': 0,
        'Flexor': 0,
    }

    # Mediator to PC weight
    MR = 100.

    # Spike generator weight
    SG = 200.

    INH = 100


SIMULATION_TIME = 500.
