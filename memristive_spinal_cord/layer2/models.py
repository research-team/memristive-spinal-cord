from enum import Enum


class SynapseModels(Enum):
    STATIC_SYNAPSE = 'static_synapse'
    STDP_SYNAPSE = 'stdp_synapse'
    STDP_DOPA_SYNAPSE = 'stdp_dopamine_synapse'
    STDP_SERO_SYNAPSE = 'stdp_serotonin_synapse'
    STDP_NORA_SYNAPSE = 'stdp_noradrenaline_synapse'


class NeuronModels(Enum):
    IAF_PSC_EXP = 'iaf_psc_exp'
    IAF_PSC_ALPHA = 'iaf_psc_alpha'
    HH_PSC_ALPHA = 'hh_psc_alpha'
    HH_COND_EXP_TRAUB = 'hh_cond_exp_traub'
    IZHIKEVICH = 'izhikevich'


class Neurotransmitters(Enum):
    GLU = 'Glu'
    GABA = 'GABA'
