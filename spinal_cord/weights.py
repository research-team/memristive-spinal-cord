from enum import Enum


class Weights(Enum):

    # Tier Interconnections
    e0e1 = 100.
    e0e3 = 100.
    e1e2 = 90.
    e2e1 = 90.
    e2i1 = 100.
    e3e4 = 100.
    e3i0 = 100.
    e4e3 = 100.
    i0e1 = -60.
    i1e1 = -50.

    # Coonnections between tiers

    e0e0 = 0.
    e3e0 = 0.
    e2e2 = 100.

    # Connections to pool

    e2p = 10

    # Motogroup interconnections

    aff_ia_moto = 15.
    aff_ia_ia = 5.
    aff_ii_ia = 5.
    aff_ii_ii = 28.
    ii_moto = 27.
    ia_ia = -0.7
    ia_moto = -6.9

    # Pool to Motogroup
    p_ex_moto = 55.
    p_ex_ia = 5.
    p_fl_moto = 55.
    p_fl_ia = 5.
