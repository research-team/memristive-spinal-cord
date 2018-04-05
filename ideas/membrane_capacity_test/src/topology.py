import nest
import sys
sys.path.append('/home/cmen/Code/road-to-heaven/un/lab/nest_practice/')
from membrane_capacity_test.src.tools.multimeter import add_multimeter


low_capacity_neuron = nest.Create(
    model='hh_cond_exp_traub',
    n=1,
    params={'C_m': 100., 'V_m': -70., 'E_L': -70.})
middle_capacity_neuron = nest.Create(
    model='hh_cond_exp_traub',
    n=1,
    params={'C_m': 200., 'V_m': -70., 'E_L': -70.})
high_capacity_neuron = nest.Create(
    model='hh_cond_exp_traub',
    n=1,
    params={'C_m': 300., 'V_m': -70., 'E_L': -70.})

spike_generator = nest.Create(
    model='spike_generator',
    n=1,
    params={
        'spike_times': [10.],
        'spike_weights': [5.]})

# connect multimeters with neurons
nest.Connect(
    pre=add_multimeter('low_capacity'),
    post=low_capacity_neuron)
nest.Connect(
    pre=add_multimeter('middle_capacity'),
    post=middle_capacity_neuron)
nest.Connect(
    pre=add_multimeter('high_capacity'),
    post=high_capacity_neuron)


# connect spike generator with neurons
nest.Connect(
    pre=spike_generator,
    post=low_capacity_neuron)
nest.Connect(
    pre=spike_generator,
    post=middle_capacity_neuron)
nest.Connect(
    pre=spike_generator,
    post=high_capacity_neuron)
