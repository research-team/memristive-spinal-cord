## Protocol

Protocol for simulation of the reflex arc.

![Figure](memristive_reflex_arc.png)

Figure 1. 
Motoneuron "moto", inhibitory interneuron "in", fibers "ia" and "ii'. 
Fibers "ia" and "ii" connect with two generators, which rate of 300 Hz.
Fiber "ia" excites glutamine motoneuron.
And fiber "ii" excites glutamine interneuron,
which inhibits GABA motoneuron. 

![Figure](result.png) 

Figure 2. Fiber "ii" excites interneuron "in" and it inhibits motoneuron "moto". 
Hyperpolarization arises.
In this time fiber "ia" excites "moto".
Impulse arises, but doesn't reach threshold.
Next impulse from "ia" induce spike in "moto".
Then fiber "ia" once more excites "moto".
Impulse arises and again doesn't reach threshold.
But with the following excitation from "ia", spike arises.
Next 40 ms in "moto" spikes don't arise.
Only impulses, which does't reach threshold, because of inhibition by "in".
At 90 ms in "moto" spike arises due to impulse from "ia".
Time of simulation finished.