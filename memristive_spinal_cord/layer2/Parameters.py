from enum import Enum


class Weights(Enum):
    # Weights of layer2 connections between neuronal groups
    # For higher convenience variables placed correspondingly to the scheme:
    # Tier6 on the top and Tier1 on the bottom

    # Right to Left groups of the same tier connection of weights
    # RLN means (R)ight group to (L)eft group of Tier(N)

    RL6 = 0.
    RL5 = 0.
    RL4 = 0.
    RL3 = 0.
    RL2 = 0.
    RL1 = 0.

    # Left to Right groups of the same tier connection weights
    # LRN means (L)eft group to (R)ight group of Tier(N)

    LR6 = 0.
    LR5 = 0.
    LR4 = 0.
    LR3 = 0.
    LR2 = 0.
    LR1 = 0.

    # Left to Left groups of neighbour tiers connection weights
    # LLMN means (L)eft group of Tier(M) to (L)eft group of Tier(N)

    LL65 = 0.
    LL54 = 0.
    LL43 = 0.
    LL32 = 0.
    LL21 = 0.

    # Right to Right groups of neighbour tiers connections weights
    # RRMN means (R)ight group of Tier(M) to (R)ight group of Tier(N)

    RR56 = 0.
    RR45 = 0.
    RR34 = 0.
    RR23 = 0.
    RR12 = 0.

    # Right group of the higher tier to Inhibitory groups of the lower tier connection weights
    # RIMN means (R)ight group of Tier(M) to (I)nhibitory group of Tier(N)

    RI65 = 0.
    RI54 = 0.
    RI43 = 0.
    RI32 = 0.
    RI21 = 0.

    # Inhibitory group to Right group of the same tier connection weights
    # IRN means (I)nhibitory group to (R)ight group of Tier(N)

    IR5 = 0.
    IR4 = 0.
    IR3 = 0.
    IR2 = 0.
    IR1 = 0.

    # Left group to Interneuronal Pool connection weights
    # LIPN means (L)eft group of Tier(N) to (I)nterneuronal (P)ool group

    LIP6 = 0.
    LIP5 = 0.
    LIP4 = 0.
    LIP3 = 0.
    LIP2 = 0.
    LIP1 = 0.

    # Mediator neuron to Right group connection weights
    # MR means (M)ediator to (R)ight group of Tier1

    MR = 0.


class Constants(Enum):

    NEURONS_IN_GROUP = 20

    REFRACTORY_PERIOD = [1.8, 2.2]

    SIMULATION_TIME = 100.  # milliseconds
    SPIKE_GENERATOR_TIMES = [25 * i + 0.1 for i in range(int(SIMULATION_TIME // 25))]

    SYNAPTIC_DELAY_EX = [1.1, 1.3]
    SYNAPTIC_DELAY_INH = [1.1, 1.2]
