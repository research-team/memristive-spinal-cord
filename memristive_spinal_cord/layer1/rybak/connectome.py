import neucogar.namespaces as NEST_NAMESPACE
from neucogar.SynapseModel import SynapseModel

import memristive_spinal_cord.layer1.rybak.params.neucogar as layer1_params

# synapses
Glutamatergic = SynapseModel(
    "Glutamatergic",
    nest_model=NEST_NAMESPACE.STDP_SYNAPSE,
    params=layer1_params.stdp_glu
)
GABAergic = SynapseModel(
    "GABAergic",
    nest_model=NEST_NAMESPACE.STDP_SYNAPSE,
    params=layer1_params.stdp_gaba
)
