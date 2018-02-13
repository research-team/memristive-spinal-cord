# The Spinal Cord simulation

## Intro
![The diagram of the scheme](img/basic-structure.png)
This model is used to simulate the S1 segment of the spinal cord. It consists of several logical parts.
Let's go from left to right follow the data flow.

1. **Afferents Data**. The .txt files contain the experimental data gathered by
[Moraud's team](http://linkinghub.elsevier.com/retrieve/pii/S0896627316000106).
Each file represented as 60 rows respectively for 60 afferents.
Each row contains a list of frequencies, each value corresponds an interval of 20 milliseconds.
2. **Afferents**. The _Afferents_ are several classes purposed to transform the _Afferents Data_ into a list of
spike times for NEST spike generators which represent II and Ia afferents.
3. **Stimulation Data**. These datafiles contain info about how many percents of afferents take a part in
stimulation actions.
4. **EES**. Performs an 'Electric Stimulation' which acts on a specific number of afferents noted in _Stimulation Data_.
5. **Level 1**. Two grand groups of alpha-motoneurons and four interneuronal groups. Stimulated afferents acts on
both motoneuron and interneuronal groups. Links between the _Layer 1_ and the _Layer 2_ are not clear now (12.02.2018).
6. **Level 2**. The level contains 6 layers produces responses with delay between them. Also contains the
interneuronal pool which excitates a specific motoneuron groups (Extensor of Flexor) and inhibits the antagonist-group,
and zero-layer which regulates activity of unterneuronal pool.