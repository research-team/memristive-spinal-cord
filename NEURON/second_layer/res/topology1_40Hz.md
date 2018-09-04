## Topology 1

With sensory and afferent activities.

### Afferent and sensory activities (EES + 10% Afferent and sensory fibers spike)

![Aff](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membAff_40hz_v3.png)

![Sens](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membS1_40hz_v3.png)

### EES 40Hz 100% inhibitory (100% Afferent and sensory fibers spike)

#### Interneuronal pool

The frequency on sensory inputs is 60 Hz thus EES stimulus 2, 4, 6 occur during the refractory period.

In first slice, first spike is triggered by EES and second by sensory input.

In second slice, first spike is triggered by sensory input (EES stimulus occur during the refractory period), second and third spikes are triggered by second layer.

In third slice, first spike is triggered by EES stimulus, second spike is triggered by second layer and third spike by sensory input.

In fourth slice, first spike is triggered by sensory input (EES stimulus occur during the refractory period), second spike is triggered by second layer and third spike by sensory input.

In fifth slice, first spike is triggered by EES stimulus, second spike is triggered by second layer and third spike by sensory input.

In sixth slice, first spike is triggered by EES and second by second layer.

###### Extracellular potential

![exIP1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraIP40Hz.png)

###### Membrane potential

![IP1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membIP40Hz.png)

#### Motoneurons

We can notice that Motoneuron doesn't respond to all EES stimulation. It happens because of afferent and sensory fibers activities.

In first slice, first spike is triggered by EES and second by Interneuronal pool which is triggered by sensory input.

In second slice, first spike is triggered by EES and second by second layer.

In third slice, first spike is triggered by afferent activity (EES stimulus occur during the refractory period) and second by second layer.

In fourth slice, first spike is triggered by Interneuronal pool which is triggered by sensory input, second spike by second layer.

In fifth slice, first spike is triggered by EES stimulus, second spike is triggered by second layer.

In sixth slice, first spike is triggered by Interneuronal pool and afferent activity (EES stimulus occur during the refractory period) and second by second layer.

###### Extracellular potential

![exMN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraMN40Hz.png)

###### Membrane potential

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membMN40Hz.png)

#### Second layer potential

In this plot we also can see influence of sensory fibers activity on second layer.

In first slice, first activity is triggered by EES and second by sensory input.

In second slice, first activity is triggered by sensory input and second by EES.

In third slice, activity is triggered by EES.

In fourth slice, activity is triggered by sensory input.

In fifth slice, first activity is triggered by sensory input, second activity is triggered by EES stimulus.

In sixth slice, activity is triggered by sensory input.

![SL1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membSL40Hz.png)

### EES 40Hz 100% inhibitory (10% Afferent and sensory fibers spike)

Then we reduced the active afferent and sensory fibers to 10%. This led to decrease the effect of sensory fibers activity on second layer. 

#### Interneuronal pool

Because of connection between Interneuronal pool and sensory fibers EES stimulus 4, 6 occur during the refractory period.

In first slice, first spike is triggered by EES and second by sensory input.

In second slice, first spike is triggered by EES and second by second layer.

In third slice, first spike is triggered by EES and second by second layer.

In fourth slice, first spike is triggered by sensory input (EES stimulus occur during the refractory period), second spike is triggered by second layer.

In fifth slice, first spike is triggered by EES stimulus, second spike is triggered by second layer and third spike by sensory input.

In sixth slice, first spike is triggered by sensory input, second spike is triggered by EES stimulus and third spike by second layer.

##### Extracellular potential

![exIP1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraIP_40hz_v2.png)

##### Membrane potential

![IP1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membIP_40hz_v2.png)

#### Motoneurons 

In first slice, first spike is triggered by EES and second by Interneuronal pool which is triggered by sensory input.

In second slice, first spike is triggered by EES and second by second layer.

In third slice, first spike is triggered by EES and second by second layer.

In fourth slice, first spike is triggered by Interneuronal pool which is triggered by sensory input, second spike by second layer.

In fifth slice, first spike is triggered by EES stimulus, second spike is triggered by second layer.

In sixth slice, first spike is triggered by EES and second by second layer.

##### Extracellular potential

![exMN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraMN_40hz_v2.png)

##### Membrane potential

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membMN_40hz_v2.png)

#### Second layer potential

There are 4 sublayers enough for delay.

In first slice, activity is triggered by EES.

In second slice, activity is triggered by EES.

In third slice, activity is triggered by EES.

In fourth slice, activity is triggered by EES.

In fifth slice, activity is triggered by EES and sensory input thus 3th sublayer spike.

In sixth slice, activity is triggered by EES and sensory input thus 4th sublayer spike.

All subthreshold activity is triggered by sensory input.

![SL1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membSL40Hz_v2.png)

### EES 40Hz 100% inhibitory (10% Afferent and sensory fibers spike) without IP-S1 connection

We get the necessary Motoneurons activity without connection between Interneuronal pool and sensory fibers.

#### Interneuronal pool

All spikes are triggered by second layer.

##### Extracellular potential

![exIP1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraIP_40hz_v3.png)

##### Membrane potential

![IP1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membIP_40hz_v3.png)

#### Motoneurons 

In first slice, first spike is triggered by EES, second low activity by second layer and third low activity by afferent.

In second slice, first spike is triggered by EES, second low activity by second layer and third low activity by afferent.

In third slice, first spike is triggered by EES and second by second layer.

In fourth slice, first spike is triggered by EES and second by second layer.

In fifth slice, first spike is triggered by EES and second by second layer.

In sixth slice, first spike is triggered by EES and second by second layer.

##### Extracellular potential

![exMN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraMN_40hz_v3.png)

##### Membrane potential

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membMN_40hz_v3.png)

#### Second layer potential

There are 4 sublayers enough for delay.

In first slice, activity is triggered by EES.

In second slice, activity is triggered by EES.

In third slice, activity is triggered by EES.

In fourth slice, activity is triggered by EES.

In fifth slice, activity is triggered by EES and sensory input thus 3th sublayer spike.

In sixth slice, activity is triggered by EES and sensory input thus 4th sublayer spike.

All subthreshold activity is triggered by sensory input.

![SL1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membSL40Hz_v3.png)

### Only Afferent activity

All activity is triggered by Afferent.

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membMN_Aff.png)

### Without second layer (ESS, Afferents and Sensory fibers)

We can notice that delay during the 1-4 period without second layer.

In first slice, first spike is triggered by EES, second low activity by afferent and third low activity by sensory input.

In second slice, first spike is triggered by EES, second spike by afferent and sensory input.

In third slice, first spike is triggered by EES and second low activity is triggered by afferent.

In fourth slice, first spike is triggered by EES and second spike by afferent.

In fifth slice, first low activity is triggered by EES (EES stimulus occur during the motoneuron refractory period) and second spike by afferent and sensory input.

In sixth slice, first spike is triggered by EES, low activity is triggered by afferent and sensory input.

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membMN_40hz_v4.png)

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraMN_40hz_v4.png)

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/SL_Memb.png)

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/WithoutSL.png)

## Сomparison of model with second layer and model without second layer

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/membCompare.png)

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraCompare.png)

## Сomparison with experimental data

In our model, response from second level comes earlier than in the experimental data.

![MN1](https://github.com/research-team/memristive-spinal-cord/blob/master/reflex_arc/neuron/second_layer/res/extraCompare_copy.png)
 
