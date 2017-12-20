from memristive_spinal_cord.layer1.moraud.neuron_groups import Layer1Neurons
from memristive_spinal_cord.layer1.params.neuron_group_params import NeuronGroupParams

params_storage = dict()

inter_model_params = {
    'V_m': 0.0,
    'V_reset': 0.0,
    'V_th': 1.0,
    'tau_m': 30.0,
    'tau_syn_ex': 0.5,
    'tau_syn_in': 5.0
    # 't_ref': 0.0,
}
inter_model_number = 196
inter_model_type = 'iaf_psc_alpha'

params_storage[Layer1Neurons.FLEX_INTER_1A] = NeuronGroupParams(
    model=inter_model_type,
    params=inter_model_params,
    number=inter_model_number,
)

params_storage[Layer1Neurons.EXTENS_INTER_1A] = NeuronGroupParams(
    model=inter_model_type,
    params=inter_model_params,
    number=inter_model_number,
)

params_storage[Layer1Neurons.FLEX_INTER_2] = NeuronGroupParams(
    model=inter_model_type,
    params=inter_model_params,
    number=inter_model_number,
)

params_storage[Layer1Neurons.EXTENS_INTER_2] = NeuronGroupParams(
    model=inter_model_type,
    params=inter_model_params,
    number=inter_model_number,
)
