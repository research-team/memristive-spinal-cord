from enum import Enum


class TierWeights(Enum):
    # Weights of layer2 connections between neuronal groups
    # For higher convenience variables placed correspondingly to the scheme:
    # Tier6 on the top and Tier1 on the bottom

    LR = [150., 0., 0., 0., 0., 0.]  # Left group to right group
    RL = [101., 0., 0., 0., 0., 0.]  # Right group to left group
    RI = [150., 0., 0., 0., 0., 0.]  # Right group to Inhibitory group
    IL = [5000., 0., 0., 0., 0., 0.]  # Inhibitory to left group
    CR = [150., 0., 0., 0., 0., 0.] # Control to right group


    # Mediator neuron to Right group connection weights
    # MR means (M)ediator to (R)ight group of Tier1

    MR = 150.

    # Spike generator weight

    SG = 200.


class HiddenWeights(Enum):
    pass


class Constants(Enum):
    NEURONS_IN_GROUP = 20

    REFRACTORY_PERIOD = [1.8, 2.2]
    ACTION_TIME_EX = 0.5
    ACTION_TIME_INH = 5.

    SIMULATION_TIME = 150.  # milliseconds
    SPIKE_GENERATOR_TIMES = [25 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]
    SPIKE_GENERATOR_WEIGHTS = [TierWeights.SG.value for _ in SPIKE_GENERATOR_TIMES]

    SYNAPTIC_DELAY_EX = [1.1, 1.3]
    SYNAPTIC_DELAY_INH = [1.1, 1.2]

    RESOLUTION = 0.1
    LOCAL_NUM_THREADS = 2


class Paths(Enum):
    DATA_DIR_NAME = 'raw_data'
    FIGURES_DIR_NAME = 'images'
