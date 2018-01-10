from memristive_spinal_cord.layer1.moraud.neuron_groups import Layer1Neurons
from memristive_spinal_cord.layer1.params.neuron_group_params import NeuronGroupParams
from memristive_spinal_cord.layer1.moraud.afferents import Layer1Afferents
from memristive_spinal_cord.layer1.params import afferents_params

params_storage = dict()

# parameters below are for normalized 'iaf_psc_alpha' as in Neuron Simulator 'IntFire4'
# that is why V_th=1.0 and V_reset=0.0
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


motor_model_params = {
    'tau_syn_ex': 0.5,
    'tau_syn_in': 1.5,
    't_ref': 2.0, # 'tau_m': 2.5
}
motor_model_number = 169
motor_model_type = 'hh_psc_alpha'

params_storage[Layer1Neurons.EXTENS_MOTOR] = NeuronGroupParams(
    model=motor_model_type,
    params=motor_model_params,
    number=motor_model_number,
)
params_storage[Layer1Neurons.FLEX_MOTOR] = NeuronGroupParams(
    model=motor_model_type,
    params=motor_model_params,
    number=motor_model_number,
)

afferents_filepath = "/layer1/moraud/afferents_data/"
generator_number_1a = 60
params_storage[Layer1Afferents.FLEX_1A] = afferents_params.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents_params.Speed.FIFTEEN,
    interval=afferents_params.Interval.TWENTY,
    type=afferents_params.Types.ONE_A,
    muscle=afferents_params.Muscles.FLEX,
    number=generator_number_1a,
)
params_storage[Layer1Afferents.EXTENS_1A] = afferents_params.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents_params.Speed.FIFTEEN,
    interval=afferents_params.Interval.TWENTY,
    type=afferents_params.Types.ONE_A,
    muscle=afferents_params.Muscles.EXTENS,
    number=generator_number_1a,
)

generator_number_2 = 60
params_storage[Layer1Afferents.FLEX_2] = afferents_params.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents_params.Speed.FIFTEEN,
    interval=afferents_params.Interval.TWENTY,
    type=afferents_params.Types.TWO,
    muscle=afferents_params.Muscles.FLEX,
    number=generator_number_2,
)
params_storage[Layer1Afferents.EXTENS_2] = afferents_params.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents_params.Speed.FIFTEEN,
    interval=afferents_params.Interval.TWENTY,
    type=afferents_params.Types.TWO,
    muscle=afferents_params.Muscles.EXTENS,
    number=generator_number_2,
)
