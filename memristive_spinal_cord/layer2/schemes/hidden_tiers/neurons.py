from enum import Enum
from memristive_spinal_cord.layer2.schemes.hidden_tiers.parameters import Constants


class Neurons(Enum):

    NEUCOGAR = {
        't_ref': Constants.REFRACTORY_PERIOD.value,
        'V_m': -70.0,
        'E_L': -70.0,
        'E_K': -77.0,
        'g_L': 30.0,
        'g_Na': 12000.0,
        'g_K': 3600.0,
        'C_m': 134.0,  # Capacity of membrane (pF)
        'tau_syn_ex': Constants.ACTION_TIME_EX.value,
        'tau_syn_in': Constants.ACTION_TIME_INH.value
    }
