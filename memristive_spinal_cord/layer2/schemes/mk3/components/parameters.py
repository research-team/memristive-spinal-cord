from enum import Enum


class Weights(Enum):
    # Weights of layer2 connections between neuronal groups
    # For higher convenience variables placed correspondingly to the scheme:
    # Tier6 on the top and Tier1 on the bottom

    # Right to Left groups of the same tier connection of weights
    # RL[N] means (R)ight group to (L)eft group of Tier(N+1)

    RL = [
        0.,
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Left to Right groups of the same tier connection weights
    # LR[N] means (L)eft group to (R)ight group of Tier(N+1)

    LR = [
        0.,
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Left to Left groups of neighbour tiers connection weights
    # LL[N] means (L)eft group of Tier(N+2) to (L)eft group of Tier(N+1)

    LL = [
        0.,
        0.,
        0.,
        0.,
        0.
    ]

    # Right to Right groups of neighbour tiers connections weights
    # RR[N] means (R)ight group of Tier(N+1) to (R)ight group of Tier(N+2)

    CC = [
        0.,
        0.,
        0.,
        0.,
        25.
    ]

    # Right group of the higher tier to Inhibitory groups of the lower tier connection weights
    # RI[N] means (R)ight group of Tier(N+2) to (I)nhibitory group of Tier(N+1)

    RI = [
        0.,
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Inhibitory group to Left group of the same tier connection weights
    # IL[N] means (I)nhibitory group to (L)eft group of Tier(N+1)

    IL = [
        0.,
        0.,
        0.,
        0.,
        0.,
        50.
    ]

    # Control group to Right group

    CR = [
        0.,
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Left group to Interneuronal Pool connection weights
    # LIP[N] means (L)eft group of Tier(N+1) to (I)nterneuronal (P)ool group

    LIP = [
        1.,
        1.,
        1.,
        1.,
        1.,
        2.
    ]

    # Mediator neuron to Right group connection weights
    # MR means (M)ediator to (R)ight group of Tier1

    MR = 150.

    # Spike generator weight

    SG = 200.


class HiddenWeights(Enum):

    # Left hidden group to Right hidden group
    LR = [
        0.,
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Right hidden group to Left hidden group
    RL = [
        0.,
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Left hidden excitatory group to Right hidden Inhibitory group
    LI = [
        0.,
        0.,
        0.,
        0.,
        150.
    ]

    # Left hidden Inhibitory group to Left excitatory group
    IL = [
        50.,
        50.,
        50.,
        50.,
        50.
    ]

    # Right Inhibitory to Right Excitatory

    RIRE = [
        150.,
        150.,
        150.,
        150.,
        150.
    ]

    # Upper Control to Inhibitory Down

    UCID = [
        0.,
        0.,
        0.,
        0.,
        0.
    ]

    # Left to Control Up

    LCU = [
        40.,
        40.,
        40.,
        40.,
        40.
    ]

    # Control Down to Left Excitatory

    CDLE = [
        0.,
        0.,
        0.,
        0.,
        0.,
        50.
    ]

    # Left hidden Inhibitory group to Right group Down of lower tier
    IRD = [
        0.,
        0.,
        0.,
        0.,
        120.
    ]

    # From upper tier Right group to Right hidden Inhibitory group
    RID = [
        0.,
        0.,
        0.,
        0.,
        150.
    ]


class Constants(Enum):
    NEURONS_IN_GROUP = 20

    REFRACTORY_PERIOD = [1.8, 2.2]
    ACTION_TIME_EX = 0.5
    ACTION_TIME_INH = 5.

    SIMULATION_TIME = 150.  # milliseconds
    SPIKE_GENERATOR_TIMES = [25 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]
    SPIKE_GENERATOR_WEIGHTS = [Weights.SG.value for _ in SPIKE_GENERATOR_TIMES]

    SYNAPTIC_DELAY_EX = [1.1, 1.3]
    SYNAPTIC_DELAY_INH = [1.1, 1.2]

    RESOLUTION = 0.1
    LOCAL_NUM_THREADS = 2


class Paths(Enum):
    DATA_DIR_NAME = 'raw_data'
    FIGURES_DIR_NAME = 'images'
