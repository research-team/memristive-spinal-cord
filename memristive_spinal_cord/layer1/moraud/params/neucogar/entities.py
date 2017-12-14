from memristive_spinal_cord.layer1.moraud.entities import Layer1Entities
from memristive_spinal_cord.layer1.params.neuron_group_params import NeuronGroupParams
import memristive_spinal_cord.layer1.params.afferents_params as afferents

general_neuron_model = {
    # 't_ref': [2.5, 4.0],  # Refractory period
    't_ref': 2.5,  # Refractory period
    'V_m': -70.0,  #
    'E_L': -70.0,  #
    'E_K': -77.0,  #
    'g_L': 30.0,  #
    'g_Na': 12000.0,  #
    'g_K': 3600.0,  #
    'C_m': 134.0,  # Capacity of membrane (pF)
    'tau_syn_ex': 0.2,  # Time of excitatory action (ms)
    'tau_syn_in': 2.0  # Time of inhibitory action (ms)
}

neuron_number_in_group = 20

params_storage = dict()

nest_neuron_model = "hh_cond_exp_traub"

params_storage[Layer1Entities.FLEX_MOTOR] = NeuronGroupParams(
    model=nest_neuron_model,
    params=general_neuron_model,
    number=neuron_number_in_group,
)

params_storage[Layer1Entities.EXTENS_MOTOR] = NeuronGroupParams(
    model=nest_neuron_model,
    params=general_neuron_model,
    number=neuron_number_in_group,
)

params_storage[Layer1Entities.FLEX_INTER_1A] = NeuronGroupParams(
    model=nest_neuron_model,
    params=general_neuron_model,
    number=neuron_number_in_group,
)

params_storage[Layer1Entities.EXTENS_INTER_1A] = NeuronGroupParams(
    model=nest_neuron_model,
    params=general_neuron_model,
    number=neuron_number_in_group,
)

params_storage[Layer1Entities.FLEX_INTER_2] = NeuronGroupParams(
    model=nest_neuron_model,
    params=general_neuron_model,
    number=neuron_number_in_group,
)

params_storage[Layer1Entities.EXTENS_INTER_2] = NeuronGroupParams(
    model=nest_neuron_model,
    params=general_neuron_model,
    number=neuron_number_in_group,
)

afferents_filepath = "/layer1/moraud/afferents/data/"
generator_number_1a = 20
params_storage[Layer1Entities.FLEX_AFFERENT_1A] = afferents.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents.Speed.FIFTEEN,
    interval=afferents.Interval.TWENTY,
    type=afferents.Types.ONE_A,
    muscle=afferents.Muscles.FLEX,
    number=generator_number_1a,
)
params_storage[Layer1Entities.EXTENS_AFFERENT_1A] = afferents.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents.Speed.FIFTEEN,
    interval=afferents.Interval.TWENTY,
    type=afferents.Types.ONE_A,
    muscle=afferents.Muscles.EXTENS,
    number=generator_number_1a,
)

generator_number_2 = 20
params_storage[Layer1Entities.FLEX_AFFERENT_2] = afferents.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents.Speed.FIFTEEN,
    interval=afferents.Interval.TWENTY,
    type=afferents.Types.TWO,
    muscle=afferents.Muscles.FLEX,
    number=generator_number_2,
)
params_storage[Layer1Entities.EXTENS_AFFERENT_2] = afferents.AfferentsParamsFile(
    filepath=afferents_filepath,
    speed=afferents.Speed.FIFTEEN,
    interval=afferents.Interval.TWENTY,
    type=afferents.Types.TWO,
    muscle=afferents.Muscles.EXTENS,
    number=generator_number_2,
)
