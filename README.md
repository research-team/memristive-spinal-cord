
<img src="/images/logo/spinal-cord-logo.png?raw=true 'Logo'" alt="Logo" height=200/>


### About
Model of ES (Electric stimulation) of spinal cord. It should enhance walking cycle with correct parameters. A proposed scheme is below.

<img src="/images/spinal-cord-diagram/spinal-cord-diagram.png?raw=true 'Reflexes'" alt="Model diagram" height=500/>

### Install

The project is compatible with python3 so when you start working on it you better have a *python3 virtualenv*. What is a virtualenv and how to work with it? Please read [here](http://www.simononsoftware.com/virtualenv-tutorial-part-2/). After you have got a *virtualenv* tool.
```
mkvirtualenv python3.5 -p /usr/bin/python3.5
```

After that you need to go the project's root and execute the below. Those are project's dependencies. For more information please read [this](https://caremad.io/posts/2013/07/setup-vs-requirement/).
```
pip install -r requirements.txt 
```

#### Problem intro
During SCI(Spinal Cord Injury) a rat loses its ability to move hind limbs. It restores its ability to walk with ES(Epidural Stimulation) only after 5 weeks. This restoration of walking happens in sync with the appearance of LR(Late Response) in the EMG of muscles. MR(Middle Response) appears during the 2nd week. Injection of 5-HT enhances the same effect of walking restoration almost immediately when applied directly to the spinal cord. This allows us to derive that during first two weeks the monosynaptic circuit or the first level restores and it brings MR to the EMG. During five weeks the polysynaptic circuit restores and brings LR to the EMG.

#### Some biology details
Muscle is innervated by motoneurons (efferents), sensory neurons (afferents). They can be classified.

**MotoNeurons**
- Alpha motoneurons. Innervate muscles. Convey signals to contraction/stretch.
- Gamma motoneurons. Innervate muscle spindles. Convey signals to contraction/stretch.

**Sensory neurons**
- Group 1.
  - Group 1a. Innervates muscle spindles. Conveys input about contraction/stretch. Fast. It excites its agonist, inhibits antagonist.
  - Group 1b. Innervates junction between muscle and its tendon. Not presented on images. Conveys input about its own contraction/stretch. It inhibits its agonist.
- Group 2. Innervates muscle spindles. Conveys input about contraction/stretch. Slow. It excites its agonist, inhibits antagonist. When a muscle is relaxed, all of its afferents are zero. Lets contract the muscle. Information about its contraction/stretch is conveyed with 0.2s delay here comparing to Group 1a.

<img src="/images/biology-intro/reflexes.jpg?raw=true 'Reflexes'" alt="Reflexes" height=300/>

<img src="/images/biology-intro/innervation.jpg?raw=true 'Muscle innervation'" alt="Muscle innervation" height=300/>

#### About computational model of Moraud and Marco 2016
They research the influence of Group 1a, Group 2 afferents only. Their simple model is presented at the figure 1A. Afferents are represented as Frequency Generators. They generate different activity for different environments. Those activities are predefined and were gotten by recording from real rat's afferents in those different environments. Model's validation:
- Serotonin-mediated modulation of motoneurons is conveyed by reducing the conductance of potassium-calcium gated ion channels. They report that the lack of 5HT modulation can be compensated by increasing ES frequency. Figure 8A.
- Variation of single ES intensity as on the Figure 1E.
- During locomotion. Increase of ES frequency led to linear increase in the mean firing rate of Ia and II afferents comparable to the increase of single ES. Temporal profiles of afferent firing rates were preserved. The linear increase in afferents led to a linear increase of motoneurons during their active phase. Increase of ES amplitude led to direct recruitment of motoneurons and disruption of alternation between flexor and extensor. See Figure 2A, 2B. **How did the increase of ES frequency lead to the increase of afferents frequency, if afferents were represented as Frequency generators?**
- During locomotion. Protocols of ES: 40 hz + 1.2 motor threshold amplitude, 80 hz + 1.2 motor threshold amplitude, 40 hz + 1.4 motor threshold amplitude, - must behave as Figure 2C.
- During locomotion. The duration of gait cycle must be sync with the step speed as on Figure 2D.
- Each phase of gait is independent of the other e.g. we can modulate ES each phase independently in order to balance step heights for example. The collateral of this is a full balance of steps. So, independent ES frequencies are much better.

General questions:
- Temporal profile in firing rate?
- Frequency harmonics?

#### Network parameters

##### Neurons
Numbers below are per muscle e.g. for example the flexor. For the antagonist muscle numbers are the same. Initial numbers were taken from [Moraud 2016](https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=189786).

**MotoNeurons**
- Number: 169
- Type: HH model with: sodium, potassium, calcium, and potassium-calcium. For the start we would use only: sodium, potassium. 

**1A fibers**
- Number: 60
- Type: Spike Generators according to `Ia firing rate = 50 + 2*stretch + 4.3*sign(strVelocity)*|strVelocity|^0.6 + 50*EMG`

**1B fibers**
- ?

**2 fibers**
- Number 60
- Type: Spike Generators according to `II firing rate = 80 + 13.5*stretch + 20*EMG`

**Cutaneous fibers**
- ?

**1A Inhibitory InterNeurons**
- Number: 196
- Type: [IntFire4](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#IntFire4) with `taue=0.5, taui1=5, taui2=10, taum=30`

**2 Excitatory InterNeurons**
- Number: 196
- Type: [IntFire4](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#IntFire4) with `taue=0.5, taui1=5, taui2=10, taum=30`  

##### Connections
 
**flexor 1A-fibers - flexor MotoNeurons**

Each motoneuron connected to all 1A fibers without any tricks and randomness.
- from *SynFlexFlex.hoc:53-66*, *M_Cell.hoc:105-125*
- nest connection type: all-to-all.
- type: [ExpSyn](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#ExpSyn) with `e=0, tau=0.5`.
- delay: 2 + Normal(0, 0.3 || 0.03) ms. In the paper 0.3 but in src 0.03. `Ia+taur.normal(0,0.03)`.
- weight: 0.052608, `hi_motor_S = 0.0411 + 0.0411 * 0.28`.

**flexor 1A-fibers - flexor 1A-Inhibitory InterNeurons**

Connect each interneuron 62 times with all fibers randomly picking the fiber each connection.
- from *SynFlexFlex.hoc:68-78*
- nest connection type: fixed-outdegree 62 `FromAll_IAf_ToOne_IAint`. InterNeurons are 'out'. Uniform.
- delay: 2 + Normal(0, 0.03) ms. `Ia+taur.normal(0,0.03)`
- weight: 0.0175, `hi= 0.0175`

**flexor 2-Excitatory InterNeurons - flexor MotoNeurons**

- from *SynFlexFlex.hoc:81-92*, *M_Cell.hoc:128-140*
- nest connection type: fixed-outdegree 116 `FromAll_EXIN_ToOne_MN`. MotoNeurons are 'out'. Uniform.
- type: [ExpSyn](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#ExpSyn) with `e=0, tau=0.5`.   
- delay: 1ms, `tausyn=1`
- weight: 0.00907, `hi_motor_Ex= (hi_motor_S*FromAll_IAf_ToOne_MN/FromAll_EXIN_ToOne_MN)/3`

**flexor 2-Fibers - flexor 2-Excitatory InterNeurons**

- from *SynFlexFlex.hoc:95-106*
- nest connection type: fixed-outdegree 62 `FromAll_IIf_ToOne_EXIN`. InterNeurons are 'out'. Uniform.
- delay: 3 + Normal(0, 0.03) ms, `II+taur.normal(0,0.03)`
- weight: 0.0175, `hi= 0.0175`

**flexor 2-Fibers - flexor 1A-Inhibitory InterNeurons**

- from *SynFlexFlex.hoc:108-119*
- nest connection type: fixed-outdegree 62 `FromAll_IIf_ToOne_IAint`. InterNeurons are 'out'. Uniform.
- delay: 3 + Normal(0, 0.03) ms, `II+taur.normal(0,0.03)`
- weight: 0.0175, `hi= 0.0175`

**extensor 1A-Inhibitory InterNeurons - flexor 1A-Inhibitory InterNeurons**

- from *SynExtFlex.hoc:6-18*
- nest connection type: fixed-outdegree 100 `FromAll_IAint_ToOne_IAint`. Flexor InterNeurons are 'out'. Uniform.
- delay: 1ms, `tausyn=1`
- weight: -0.007, `-(lo=0.007)`

**flexor 1A-Inhibitory InterNeurons - extensor 1A-Inhibitory InterNeurons**

- from *SynExtFlex.hoc:20-30*
- nest connection type: fixed-outdegree 100 `FromAll_IAint_ToOne_IAint`. Extensor InterNeurons are 'out'. Uniform.
- delay: 1ms, `tausyn=1`. **Warning, in the sources this delay is missing**
- weight: -0.007, `-(lo=0.007)`

**extensor 1A-Inhibitory InterNeurons - flexor MotoNeurons**

Every motoneuron connected 232 times randomly to the pool of interneurons. There is a very small possibility that it will connect 232 times to the same interneuron.
- from *SynExtFlex.hoc:32-44*, *M_Cell.hoc:141-147*
- nest connection type: fixed-outdegree 232 `FromAll_IAint_ToOne_MN`. MotoNeurons are 'out'. Uniform.
- type: [Exp2Syn](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#Exp2Syn) with `e=-75, tau1=1.5, tau2=2`.   
- delay: 1ms, `tausyn=1`
- weight: 0.0023, `lom=0.0023`

**flexor 1A-Inhibitory InterNeurons - extensor MotoNeurons**

- from *SynExtFlex.hoc:46-57*, *M_Cell.hoc:141-147*
- nest connection type: fixed-outdegree 232 `FromAll_IAint_ToOne_MN`. MotoNeurons are 'out'. Uniform.
- type: [Exp2Syn](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/mech.html#Exp2Syn) with `e=-75, tau1=1.5, tau2=2`.   
- delay: 1ms, `tausyn=1`
- weight: 0.0023, `lom=0.0023`

