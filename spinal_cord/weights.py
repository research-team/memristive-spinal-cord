from enum import Enum


def init(weights: list):
    weights = [float(weight) for weight in weights]
    Weights.aff_ia_moto = weights[0]
    Weights.aff_ia_ia = weights[1]
    Weights.aff_ii_ii = weights[2]
    Weights.ii_moto = weights[3]
    Weights.ia_ia = weights[4]
    Weights.ia_moto = weights[5]


class Weights:

    # Tier Interconnections
    e0e1 = 100.
    e0e3 = 100.
    e1e2 = 100.
    e2e1 = 100.
    e2i1 = 100.
    e3e4 = 100.
    e3i0 = 100.
    e4e3 = 100.
    i0e1 = -60.
    i1e1 = -50.
    e3i0e1 = -100.
    e2i1e1 = -50.

    # Coonnections between tiers

    e0e0 = 45.
    e3e0 = 40.
    e2e2 = 100.

    # Connections to pool

    e2p = 20.

    # Motogroup interconnections

    aff_ia_moto = 10.
    aff_ia_ia = 0.
    aff_ii_ia = 0.
    aff_ii_ii = 0.
    ii_moto = 0.
    ia_ia = 0.
    ia_moto = 0.

    # Pool to Motogroup
    p_ex_moto = 90.
    p_ex_ia = 0.
    p_fl_moto = 0.
    p_fl_ia = 0.

    # Pool interconnections

    p_extens_sus_extens_ex = 0.
    p_flex_sus_flex_ex = 0.
    p_extens_sus_flex_in = 0.
    p_flex_sus_extens_in = 0.
    p_flex_extens_in = 0.
    p_extens_flex_in = 0.
