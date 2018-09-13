#ifndef IZHIKEVICHGPU_NEURON_H
#define IZHIKEVICHGPU_NEURON_H

//#include <openacc.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <utility>
#include "Synapse.h"

using namespace std;

extern const float T_sim;
class Neuron {
    /*/// Synapse structure
    struct Synapse {
        Neuron* post_neuron{}; // post neuron ID
        int syn_delay{};		 // [steps] synaptic delay. Converts from ms to steps
        float weight{};		 // [pA] synaptic weight
        int timer{};			 // [steps] changeable value of synaptic delay
        float changing_weight{};
        int stdp_timer{};

        Synapse() = default;
        Synapse(Neuron* post, float delay, float w) {
            this-> post_neuron = post;
            this-> syn_delay = ms_to_step(delay);
            this-> weight = w;
            this-> timer = -1;
            this-> changing_weight = w;
            this-> stdp_timer = -1;
        }
    };*/
private:
    /// Object variables
    int id{};								// neuron ID
    float *spike_times{};					// array of spike time
    float *membrane_potential{};			// array of membrane potential values
    double *weights{};			// array of membrane potential values
    float *I_potential{};					// array of I
    int last_spike_time{};

    int mm_record_step = ms_to_step(0.1f); 	// step of recording membrane potential
    int iterSpikesArray = 0;				// current index of array of the spikes
    int iterVoltageArray = 0; 				// current index of array of the V_m
    int simulation_iter = 0;		        // current simulation step
    bool hasMultimeter = false;				// if neuron has multimeter
    bool hasSpikedetector = false;			// if neuron has spikedetector
    bool hasGenerator = false;				// if neuron has generator
    bool hasWeightrecorder = false;

    /// Stuff variables
    const float I_tau = 3.0f;				                            // step of I decreasing/increasing
    static constexpr float ms_in_1step = 0.1f;	                        // how much milliseconds in 1 step
    static const int steps_in_1ms = static_cast<int>(1 / ms_in_1step);  // how much steps in 1 ms
    const int STDP_TIMER = 30 * steps_in_1ms;

    /// Parameters (const)
    const float C = 100.0f;			// [pF] membrane capacitance
    const float V_rest = -72.0f;	// [mV] resting membrane potential
    const float V_th = -55.0f;		// [mV] spike threshold
    const float k = 0.7f;			// [pA * mV-1] constant ("1/R")
    const float a = 0.03f;			// [ms-1] time scale of the recovery variable U_m
    const float b = -2.0f;			// [pA * mV-1]  sensitivity of U_m to the sub-threshold fluctuations of the V_m
    const float c = -80.0f;			// [mV] after-spike reset value of V_m
    const float d = 100.0f;			// [pA] after-spike reset value of U_m
    const float V_peak = 35.0f;		// [mV] spike cutoff value
    int ref_t{}; 					// [step] refractory period

    /// State (changable)
    float V_m = V_rest;		// [mV] membrane potential
    float U_m = 0.0f;		// [pA] membrane potential recovery variable
    float I = 0.0f;			// [pA] input current
    float V_old = V_m;		// [mV] previous value for the V_m
    float U_old = U_m;		// [pA] previous value for the U_m
    float current_ref_t = 0;

public:
    Synapse* synapses = new Synapse[100];	// array of synapses
    int num_synapses{0};                    // current number of synapses (neighbors)
    int synapse_array_capacity = 100;       // length of the array of a synapses
    Synapse* input_syns = new Synapse[100];
    int num_input_syns{0};

    char* name{};

    Neuron() = default;

    Neuron(int id, float ref_t) {
        this->id = id;
        this->ref_t = ms_to_step(ref_t);
    }

    void changeCurrent(float I) {
        if (!hasGenerator) {
            this->I += I;
        }
    }

    void addMultimeter() {
        // set flag that this neuron has the multimeter
        hasMultimeter = true;
        // allocate memory for recording V_m
        membrane_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];
        // allocate memory for recording I
        I_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];
        weights = new double[ ms_to_step(T_sim) / mm_record_step ];

    }

    void addSpikedetector() {
        // set flag that this neuron has the multimeter
        hasSpikedetector = true;
        // allocate memory for recording spikes
        spike_times = new float[ ms_to_step(T_sim) / this->ref_t ];
    }

    void addGenerator(float I) {
        hasGenerator = true;
        this->I = I;
    }

    bool withMultimeter() {
        return hasMultimeter;
    }

    bool withSpikedetector() {
        return hasSpikedetector;
    }

    float step_to_ms(int step) {
        return step * ms_in_1step;  // convert steps to milliseconds
    }

    static int ms_to_step(float ms) {
        return (int) (ms * steps_in_1ms);   // convert milliseconds to step
    }

    Neuron* getThis() {
        return this;
    }

    int getID() {
        return this->id;
    }

    float* getSpikes() {
        return spike_times;
    }

    float* getVoltage() {
        return membrane_potential;
    }

    float* getCurrents() {
        return I_potential;
    }

    double* getWeights() {
        return weights;
    }

    int getVoltageArraySize() {
        return (ms_to_step(T_sim) / mm_record_step);
    }

    int getSimulationIter() {
        return simulation_iter;
    }

    int getIterSpikesArray() {
        return iterSpikesArray;
    }

    //#pragma acc routine vector
    /// Invoked every simulation step, update the neuron state
    void update_state() {
        if (current_ref_t > 0) {
            // calculate V_m and U_m WITHOUT synaptic weight
            // (absolute refractory period)
            V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old) / C;
            U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);

        } else {
            // calculate V_m and U_m WITH synaptic weight
            // (action potential)
            V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old + I) / C;
            U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);
        }

        // save the V_m and I value every mm_record_step if hasMultimeter
        if (hasMultimeter && simulation_iter % mm_record_step == 0) {
            membrane_potential[iterVoltageArray] = V_m;
            I_potential[iterVoltageArray] = I;
            if (this->id == 1)
                weights[iterVoltageArray] = synapses[0].changing_weight;
            iterVoltageArray++;
        }

        if (V_m < c)
            V_m = c;

        // threshold crossing (spike)
        if (V_m >= V_peak) {
            // set timers for all neuron synapses
            for (int i = 0; i < num_synapses; i++) {
                synapses[i].timer = synapses[i].syn_delay;
                synapses[i].stdp_timer = STDP_TIMER;
            }

            // redefine V_old and U_old
            V_old = c;
            U_old += d;

            // save spike time if hasSpikedetector
            if (hasSpikedetector) {
                printf("OOOOOOOOOOOOOO %d \n", iterSpikesArray);
                spike_times[iterSpikesArray] = step_to_ms(simulation_iter);
                iterSpikesArray++;
            }

            // set the refractory period
            current_ref_t = ref_t;
        } else {
            // redefine V_old and U_old
            V_old = V_m;
            U_old = U_m;
        }

        // update timers in all neuron synapses
        for (int i = 0; i < num_synapses; i++ ) {
            Synapse* syn = synapses + i;
            // "send spike"
            if (syn->timer == 0) {
                syn->changing_weight += updateSTDP(syn);
                updateInputSyns();
                // change the I (currents) of the post neuron

                if (syn->weight != syn->changing_weight) {
                    printf("iter = %d|||(%d) and (%d) are connected with weight = %f; ch_w = %f\n",
                           simulation_iter, this->getID(), syn->post_neuron->getID(),
                           syn->weight, syn->changing_weight);
                }

                //syn->post_neuron->changeCurrent(simulation_iter / 10);
                syn->post_neuron->changeCurrent(syn->weight);

                // set timer to -1 (thats mean no need to update timer in future without spikes)
                syn->timer = -1;
            }

            //if (syn->stdp_timer == 0) {
            //    syn->weight = syn->changing_weight;
            //    syn->stdp_timer = STDP_TIMER;
            //}

            // decrement timers
            //printf("%d %f \n", this->id, syn->timer);
            if (syn->timer > 0) {
                printf("ID %d , %d \n", this->id, syn->timer);
                syn->timer--;
            }
            if (syn->stdp_timer > 0) {
                syn->stdp_timer--;
            }
        }
        if (V_m >= V_peak)
            last_spike_time = simulation_iter;




        // update I (currents) of the neuron
        // doesn't change the I of generator neurons!!!
        if (!hasGenerator && I != 0) {
            if (I > 0) {	// turn the current to 0 by I_tau step
                I -= I_tau;	// decrease I
                if (I < 0)	// avoid the near value to 0
                    I = 0;
            } else {
                I += I_tau; // increase I
                if (I > 0)	// avoid the near value to 0
                    I = 0;
            }
        }

        // update the refractory period timer
        if (current_ref_t > 0)
            current_ref_t--;

        // update the simulation iteration
        simulation_iter++;
    }

    // STDP function (learning window)
    float W(float delta) {
        int maximal = 2;
        int minimal = -maximal;

        float result = 0.0;
        float Aplus = 1;
        float Aminus = 1;
        int Tplus = 20;
        int Tminus = 20;
        if (delta > 20 || delta < -20)
            return result;
        if (delta >= 0) {
            result += Aplus * exp(delta/Tplus);
        } else {
            result += -Aminus * exp(delta/Tminus);
        }
        printf("delta %f \n", delta);
        if (result > maximal)
            result = maximal;
        if (result < minimal)
            result = minimal;
        return result;
    }

    float updateSTDP(Synapse* syn) {
        float change = 0.0;

        printf("TTTT pre ID %d (%d)  post ID %d (%d) Weight %f\n",
               this->id,
               iterSpikesArray,
               syn->post_neuron->id,
               syn->post_neuron->getIterSpikesArray(), syn->weight);

        change += W(step_to_ms(syn->post_neuron->last_spike_time - this->last_spike_time));
        if (syn->changing_weight * (syn->changing_weight + change) < 0) {
            change = -syn->changing_weight;
        }
        printf("STDP CHANGE %f \n", change);
        return change;
    }

    void connectWith(Neuron* pre_neuron, Neuron* post_neuron, float syn_delay, float weight) {
        /// adding the new synapse to the neuron
        Synapse* syn = new Synapse(pre_neuron, post_neuron, syn_delay, weight);
        pre_neuron->synapses[pre_neuron->num_synapses++] = *syn;
        post_neuron->input_syns[post_neuron->num_input_syns++] = *syn;

        // increase array size if near to the limit
        /*
        if (num_synapses == synapse_array_capacity) {
            int new_neighbors_capacity = static_cast<int>(synapse_array_capacity * 1.5 + 1);
            auto* new_neighbors = new Synapse[new_neighbors_capacity];
            // copying
            for (int i = 0; i < num_synapses; ++i) {
                new_neighbors[i] = synapses[i];
            }
            // change the links
            synapses = new_neighbors;
            synapse_array_capacity = new_neighbors_capacity;
            delete[] new_neighbors;
        }*/

        /*printf("Connected %d (%p) to %d (%p) (%.2f) \n",
               this->getID(), this, post_neuron->getID(), post_neuron->getThis(), weight);*/
    }

    void updateInputSyns() {
        for (int i = 0; i < num_input_syns; ++i) {
            printf("___________weight = %f\n", input_syns[i].changing_weight);
            printf("-----------from post(%d) to pre(%d)\n", input_syns[i].post_neuron->getID(), input_syns[i].pre_neuron->getID());
            for (int j = 0; j < input_syns[i].pre_neuron->num_synapses; ++j) {
                if (input_syns[i].pre_neuron->synapses[j].post_neuron == this) {
                    input_syns[i].pre_neuron->synapses[j].changing_weight += 10;
                }
            }
            //input_syns[i].changing_weight += 10;
            printf("___________weight = %f\n", input_syns[i].changing_weight);
        }
    }

    ~Neuron() {
//#pragma acc exit data delete(this)
        if (hasSpikedetector) {
            delete[] spike_times;
        }

        if (hasMultimeter) {
            delete[] membrane_potential;
            delete[] I_potential;
        }
        delete[] synapses;
    }
};

#endif //IZHIKEVICHGPU_NEURON_H
