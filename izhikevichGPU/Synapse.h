#include <openacc.h>
#include "Neuron.h"

using namespace std;

class Neuron;
extern const float T_sim;

/// Synapse structure
struct Synapse {
    Neuron* pre_neuron{};
    Neuron* post_neuron{};          // post neuron ID
    int syn_delay{};		        // [steps] synaptic delay. Converts from ms to steps
    float weight{};		            // [pA] synaptic weight
    int timer{};			        // [steps] changeable value of synaptic delay
    float changing_weight{};
    int stdp_timer{};
    static constexpr float ms_in_1step = 0.1f;
    static const int steps_in_1ms = static_cast<int>(1 / ms_in_1step);

    Synapse() = default;
    Synapse(Neuron* pre, Neuron* post, float delay, float w) {
        this-> pre_neuron = pre;
        this-> post_neuron = post;
        this-> syn_delay = ms_to_step(delay);
        this-> weight = w;
        this-> timer = -1;
        this-> changing_weight = w;
        this-> stdp_timer = -1;
    }

    int ms_to_step(float ms) {
        return (int) (ms * steps_in_1ms);   // convert milliseconds to step
    }
};
