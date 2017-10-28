### About
Model of ES (Electric stimulation) of spinal cord. It should enhance walking cycle with correct parameters. A proposed scheme is below.

<img src="/diagrams/spinal-cord-diagram/spinal-cord-diagram.png?raw=true 'Reflexes'" alt="Model diagram" height=500/>

#### Problem intro
During SCI(Spinal Cord Injury) a rat loses its ability to move hind limbs. It restores its ability to walk with ES(Epidural Stimulation) only after 5 weeks. This restoration of walking happens in sync with the appearance of LR(Late Response) in the EMG of muscles. MR(Middle Response) appears during the 2nd week. Injection of 5-HT enhances the same effect of walking restoration almost immediately when applied directly to the spinal cord. This allows us to derive that during first two weeks the monosynaptic circuit or the first level restores and it brings MR to the EMG. During five weeks the polysynaptic circuit restores and brings LR to the EMG.

#### Some biology details
Muscle is innervated by motoneurons (efferents), sensory neurons (afferents). They can be classified.

**Motoneurons**
- Alpha motoneurons. Innervate muscles. Convey signals to contraction/stretch.
- Gamma motoneurons. Innervate muscle spindles. Convey signals to contraction/stretch.

**Sensory neurons**
- Group 1.
  - Group 1a. Innervates muscle spindles. Conveys input about contraction/stretch. Fast. It excites its agonist, inhibits antagonist.
  - Group 1b. Innervates junction between muscle and its tendon. Not presented on images. Conveys input about its own contraction/stretch. It inhibits its agonist.
- Group 2. Innervates muscle spindles. Conveys input about contraction/stretch. Slow. It excites its agonist, inhibits antagonist. When a muscle is relaxed, all of its afferents are zero. Lets contract the muscle. Information about its contraction/stretch is conveyed with 0.2s delay here comparing to Group 1a.

<img src="/diagrams/biology-intro/reflexes.jpg?raw=true 'Reflexes'" alt="Reflexes" height=300/>

<img src="/diagrams/biology-intro/innervation.jpg?raw=true 'Muscle innervation'" alt="Muscle innervation" height=300/>

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
