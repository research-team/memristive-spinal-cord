#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <random>
#include <curand_kernel.h>
#include <chrono>
#include <math.h>
// for file writing
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>
// my classes
#include "Group.cpp"
// colors
#define COLOR_RED "\x1b[1;31m"
#define COLOR_GREEN "\x1b[1;32m"
#define COLOR_RESET "\x1b[0m"
// IDE definitions
#ifdef __JETBRAINS_IDE__
#define __host__
#define __global__
#endif

/**
 6 cm/s = 125 [ms] has 30 slices
15 cm/s = 50 [ms] has 15 slices
21 cm/s = 25 [ms] has 6 slices

References:
  [1] https://en.wikipedia.org/wiki/Hodgkin-Huxley_model
**/

using namespace std;

unsigned int global_id = 0;
unsigned int SIM_TIME_IN_STEPS;
const int LEG_STEPS = 1;             // [step] number of full cycle steps
const float SIM_STEP = 0.025;        // [s] simulation step

// stuff variables
const int neurons_in_ip = 196;       // number of neurons in interneuronal pool
const int neurons_in_aff_ip = 196;   // number of neurons in interneuronal pool
const int neurons_in_group = 50;     // number of neurons in a group
const int neurons_in_afferent = 120; // number of neurons in afferent

const int neurons_in_moto_extensor = 210;
const int neurons_in_moto_flexor = 180;

// neuron parameters
const float E_Na = 50.0;             // [mV] Reversal potential for the Sodium current
const float E_K = -80.0;             // [mV] Reversal potential for the Potassium current
const float E_L = -72.0;             // [mV] Reversal potential for the leak current
const float E_ex = 0.0;              // [mV] Reversal potential for excitatory input
const float E_in = -80.0;            // [mV] Reversal potential for inhibitory input
const float tau_syn_exc = 0.2;       // [ms] Decay time of excitatory synaptic current (ms)
const float tau_syn_inh = 2.0;       // [ms] Decay time of inhibitory synaptic current (ms)
const float V_adj = -63.0;           // adjusts threshold to around -50 mV
const float g_bar = 1500;            // [nS] the maximal possible conductivity

// struct for human-readable initialization of connectomes
struct SynapseMetadata {
    unsigned int pre_id;         // pre neuron ID
    unsigned int post_id;        // post neuron ID
    unsigned int synapse_delay;  // [step] synaptic delay of the synapse (axonal delay is included to this delay)
    float synapse_weight;        // [nS] synaptic weight. Interpreted as changing conductivity of neuron membrane

    SynapseMetadata(int pre_id, int post_id, float synapse_delay, float synapse_weight){
        this->pre_id = pre_id;
        this->post_id = post_id;
        this->synapse_delay = static_cast<int>(synapse_delay * (1 / SIM_STEP) + 0.5);  // round
        this->synapse_weight = synapse_weight;
    }
};

// struct for human-readable initialization of connectomes
struct GroupMetadata {
    Group group;
    float* g_exc;                // [nS] array of excitatory conductivity
    float* g_inh;                // [nS] array of inhibition conductivity
    float* voltage_array;        // [mV] array of membrane potential
    vector<float> spike_vector;  // [ms] spike times

    explicit GroupMetadata(Group group){
        this->group = move(group);
        voltage_array = new float[SIM_TIME_IN_STEPS];
        g_exc = new float[SIM_TIME_IN_STEPS];
        g_inh = new float[SIM_TIME_IN_STEPS];
    }
};

vector<GroupMetadata> all_groups;
vector<SynapseMetadata> all_synapses;

// form structs of neurons global ID and groups name
Group form_group(const string& group_name, int nrns_in_group = neurons_in_group) {
    Group group = Group();
    group.group_name = group_name;     // name of a neurons group
    group.id_start = global_id;        // first ID in the group
    group.id_end = global_id + nrns_in_group - 1;  // the latest ID in the group
    group.group_size = nrns_in_group;  // size of the neurons group

    all_groups.emplace_back(group);

    global_id += nrns_in_group;
    printf("Formed %s IDs [%d ... %d] = %d\n",
           group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

    return group;
}

__host__
int ms_to_step(float ms) { return (int)(ms / SIM_STEP); }

__host__
float step_to_ms(int step) { return step * SIM_STEP; }

__global__
void neurons_kernel(const float *C_m,
                    float *V_m,
                    float *h,
                    float *m,
                    float *n,
                    float *g_exc,
                    float *g_inh,
                    const float *nrn_threshold,
                    bool *has_spike,
                    const int *nrn_ref_time,
                    int *nrn_ref_time_timer,
                    const int neurons_number,
                    int shifted_sim_iter,
                    const int activated_C_,
                    const int early_activated_C_,
                    const int sim_iter,
                    const int *begin_C_spiking,
                    const int *end_C_spiking,
                    const int decrease_lvl_Ia_spikes,
                    const int ees_spike_each_step,
                    float *g_Na,
                    float *g_K,
                    float *g_L) {
    /**
     *
     */
    // get ID of the thread
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ia aff extensor/flexor IDs [1947 ... 2186], control spike number of Ia afferent by resetting neuron current
    if (1947 <= tid && tid <= 2186) {
        // rule for the 2nd level
        if (decrease_lvl_Ia_spikes == 1 && tid % 3 == 0) {
            // reset current of 1/3 of neurons
            g_exc[tid] = 0;  // set maximal inhibitory conductivity
        } else {
            // rule for the 3rd level
            if (decrease_lvl_Ia_spikes == 2 && tid % 2 == 0) {
                // reset current of 1/2 of neurons
                g_exc[tid] = 0;  // set maximal inhibitory conductivity
            }
        }
    }

    // generating spikes for EES
    if (tid <= 50 && (sim_iter % ees_spike_each_step == 0)) {
        g_exc[tid] = g_bar;  // set spike state
    }

    __syncthreads();

    // ignore threads which ID is greater than neurons number
    if (tid < neurons_number) {
        // reset spike flag of the current neuron before calculations
        has_spike[tid] = false;

        if (activated_C_ == 0 && early_activated_C_ == 0 && 3267 <= tid && tid <= 3462 && (sim_iter % 10 == 0)) {
            curandState localState;
            curand_init(sim_iter, tid, 0, &localState);
            // normal distribution of choosing which tid's neuron will spike at each 10th step of simulation (0.25ms)
            has_spike[3267 + static_cast<int>(196 * curand_uniform(&localState))] = true;
        }
        // Skin stimulations
        if (activated_C_ == 1) {
            // Each thread gets same seed, a different sequence number, no offset
            curandState localState;
            curand_init(sim_iter, tid, 0, &localState);

            // Noise on MNs
            // [1557 ... 1946]
            if (1557 <= tid && tid <= 1946) {
                if (curand_uniform(&localState) >= 0.5)
                    V_m[tid] += curand_uniform(&localState) * 2;
                else
                    V_m[tid] -= curand_uniform(&localState) * 2;
            }

            // CV1
            if (tid == 300 && begin_C_spiking[0] < shifted_sim_iter && shifted_sim_iter < end_C_spiking[0] && curand_uniform(&localState) >= 0.5) {
                has_spike[tid] = true;
            }
            // CV2
            if (tid == 301 && begin_C_spiking[1] < shifted_sim_iter && shifted_sim_iter < end_C_spiking[1] && curand_uniform(&localState) >= 0.5) {
                has_spike[tid] = true;
            }
            // CV3
            if (tid == 302 && begin_C_spiking[2] < shifted_sim_iter && shifted_sim_iter < end_C_spiking[2] && curand_uniform(&localState) >= 0.5) {
                has_spike[tid] = true;
            }
            // CV4
            if (tid == 303 && begin_C_spiking[3] < shifted_sim_iter && shifted_sim_iter < end_C_spiking[3] && curand_uniform(&localState) >= 0.5) {
                has_spike[tid] = true;
            }
            // CV5
            if (tid == 304 && begin_C_spiking[4] < shifted_sim_iter && shifted_sim_iter < end_C_spiking[4] && curand_uniform(&localState) >= 0.5) {
                has_spike[tid] = true;
            }
        }

        // the maximal value of input current
        if (g_exc[tid] > g_bar)
            g_exc[tid] = g_bar;
        if (g_inh[tid] > g_bar)
            g_inh[tid] = g_bar;

        // use temporary V variable as V_m with adjust
        const float V = V_m[tid] - V_adj;

        // transition rates between open and closed states of the potassium channels
        float alpha_n = 0.032 * (15.0 - V) / (exp((15.0 - V) / 5.0) - 1.0);
        if (isnan(alpha_n))
            alpha_n = 0;
        float beta_n = 0.5 * exp((10.0 - V) / 40.0);
        if (isnan(beta_n))
            beta_n = 0;

        // transition rates between open and closed states of the activation of sodium channels
        float alpha_m = 0.32 * (13.0 - V) / (exp((13.0 - V) / 4.0) - 1.0);
        if (isnan(alpha_m))
            alpha_m = 0;
        float beta_m = 0.28 * (V - 40.0) / (exp((V - 40.0) / 5.0) - 1.0);
        if (isnan(beta_m))
            beta_m = 0;

        // transition rates between open and closed states of the inactivation of sodium channels
        float alpha_h = 0.128 * exp((17.0 - V) / 18.0);
        if (isnan(alpha_h))
            alpha_h = 0;
        float beta_h = 4.0 / (1.0 + exp((40.0 - V) / 5.0));
        if (isnan(beta_h))
            beta_h = 0;

        // re-calculate activation variables
        n[tid] += (alpha_n - (alpha_n + beta_n) * n[tid]) * SIM_STEP;
        m[tid] += (alpha_m - (alpha_m + beta_m) * m[tid]) * SIM_STEP;
        h[tid] += (alpha_h - (alpha_h + beta_h) * h[tid]) * SIM_STEP;

        // ionic currents
        float I_NA = g_Na[tid] * pow(m[tid], 3) * h[tid] * (V_m[tid] - E_Na);
        float I_K = g_K[tid] * pow(n[tid], 4) * (V_m[tid] - E_K);
        float I_L = g_L[tid] * (V_m[tid] - E_L);
        float I_syn_exc = g_exc[tid] * (V_m[tid] - E_ex);
        float I_syn_inh = g_inh[tid] * (V_m[tid] - E_in);

        // if neuron in the refractory state -- ignore synaptic inputs. Re-calculate membrane potential
        if (nrn_ref_time_timer[tid] > 0) {
            V_m[tid] += -(I_L + I_K + I_NA) / C_m[tid] * SIM_STEP;
        } else {
            V_m[tid] += -(I_L + I_K + I_NA + I_syn_exc + 4 * I_syn_inh) / C_m[tid] * SIM_STEP;
        }

        // re-calculate conductance
        g_exc[tid] += -g_exc[tid] / tau_syn_exc * SIM_STEP;
        g_inh[tid] += -g_inh[tid] / tau_syn_inh * SIM_STEP;

        if (V_m[tid] > E_Na)
            V_m[tid] = E_Na;
        if (V_m[tid] < -100)
            V_m[tid] = -100;

        // (threshold && not in refractory period)
        if ((V_m[tid] >= nrn_threshold[tid]) && (nrn_ref_time_timer[tid] == 0)) {
            has_spike[tid] = true;  // set spike state. It will be used in the "synapses_kernel"
            nrn_ref_time_timer[tid] = nrn_ref_time[tid];  // set the refractory period
        }

        // update the refractory period timer
        if (nrn_ref_time_timer[tid] > 0)
            nrn_ref_time_timer[tid]--;
    }
}

__global__
void synapses_kernel(const bool *neuron_has_spike,     // array of bools -- is neuron has spike or not
                     float *neuron_g_exc,              // array of excitatory conductivity per neuron (changable)
                     float *neuron_g_inh,              // array of inhibitory conductivity per neuron (changable)
                     const int *synapses_pre_nrn_id,   // array of pre neurons ID per synapse
                     const int *synapses_post_nrn_id,  // array of post neurons ID per synapse
                     const int *synapses_delay,        // array of synaptic delay per synapse
                     int *synapses_delay_timer,        // array as above but changable
                     const float *synapses_weight,     // array of synaptic weight per synapse
                     const int syn_number) {            // number of synapses
    /**
     *
     */
    // get ID of the thread
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ignore threads which ID is greater than neurons number
    if (tid < syn_number) {
        // add synaptic delay if neuron has spike
        if (synapses_delay_timer[tid] == -1 && neuron_has_spike[synapses_pre_nrn_id[tid]]) {
            synapses_delay_timer[tid] = synapses_delay[tid];
        }
        // if synaptic delay is zero it means the time when synapse increase I by synaptic weight
        if (synapses_delay_timer[tid] == 0) {
            // post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
            if (synapses_weight[tid] >= 0) {
                atomicAdd(&neuron_g_exc[synapses_post_nrn_id[tid]], synapses_weight[tid]);
            } else {
                // remove negative sign
                atomicAdd(&neuron_g_inh[synapses_post_nrn_id[tid]], -synapses_weight[tid]);
            }
            // make synapse timer a "free" for next spikes
            synapses_delay_timer[tid] = -1;
        }
        // update synapse delay timer
        if (synapses_delay_timer[tid] > 0) {
            synapses_delay_timer[tid]--;
        }
    }
}

void connect_one_to_all(const Group& pre_neurons,
                        const Group& post_neurons,
                        float syn_delay,
                        float weight) {
    /**
     *
     */
    // Seed with a real random value, if available
    random_device r;
    default_random_engine generator(r());
    normal_distribution<float> delay_distr(syn_delay, syn_delay / 5);
    normal_distribution<float> weight_distr(weight, weight / 10);

    for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
        for (unsigned int post_id = post_neurons.id_start; post_id <= post_neurons.id_end; post_id++) {
            all_synapses.emplace_back(pre_id, post_id, delay_distr(generator), weight_distr(generator));
        }
    }

    printf("Connect %s to %s [one_to_all] (1:%d). Total: %d W=%.2f, D=%.1f\n", pre_neurons.group_name.c_str(),
           post_neurons.group_name.c_str(), post_neurons.group_size, pre_neurons.group_size * post_neurons.group_size,
           weight, syn_delay);
}

float clipped_distribution(float mean, float sttdev) {
    random_device r;
    default_random_engine generator(r());
    normal_distribution<float> d_distr(mean, sttdev);

    float sigma = abs(mean) / 5;
    float probability = 0.001;
    float k = log(sqrt(2 * M_PI * probability * probability * sigma * sigma));
    float res = k < 0 ? sigma * sqrt(-2 * k) : sigma * sqrt(2 * k);
    float low = mean - res;
    float high = mean + res;

    float num = 0;
    while (true) {
        num = d_distr(generator);
        if(num >= low && num <= high) {
            return num;
        }
    }
}

void connect_fixed_outdegree(const Group& pre_neurons,
                             const Group& post_neurons,
                             float syn_delay,
                             float syn_weight,
                             int syn_degree=0) {
    // connect neurons with uniform distribution and normal distributon for syn delay and syn_weight
    random_device r;
    default_random_engine generator(r());

    // connectomes rule
    uniform_int_distribution<int> post_nrn_id_distr(post_neurons.id_start, post_neurons.id_end);
    uniform_int_distribution<int> syn_outdegree_distr((int)(neurons_in_group * 0.6), neurons_in_group);

    int outdegree = syn_degree;

    if(syn_degree == 0) {
        outdegree = syn_outdegree_distr(generator);
    }

    for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
        for (int i = 0; i < outdegree; i++) {
            int rand_post_id = post_nrn_id_distr(generator);
            // normal clipped distribution
            float syn_delay_distr = clipped_distribution(syn_delay, (syn_delay / 10));
            float syn_weight_distr = clipped_distribution(syn_weight, (syn_weight / 5));

            // check borders
            if (syn_delay_distr <= SIM_STEP)
                syn_delay_distr = SIM_STEP;

            if (syn_weight_distr >= g_bar)
                syn_weight_distr = g_bar;

            if (syn_weight_distr <= -g_bar)
                syn_weight_distr = -g_bar;
            // add meta data to the vector
            all_synapses.emplace_back(pre_id, rand_post_id, syn_delay_distr, syn_weight_distr);
        }
    }

    printf("Connect %s to %s [fixed_outdegree] (1:%d). Total: %d W=%.2f, D=%.1f\n",
           pre_neurons.group_name.c_str(), post_neurons.group_name.c_str(),
           outdegree, pre_neurons.group_size * outdegree, syn_weight, syn_delay);
}
void init_network(float E1_OM1_0,
        float E2_OM2_0,
        float E3_OM3_0,
        float E4_OM4_0,
        float E5_OM5_0,
        float CV1_OM1_0,
        float CV2_OM1_0,
        float CV3_OM1_3,
        float CV4_OM1_3,
        float CV5_OM1_3,
        float CV2_OM2_0,
        float CV3_OM2_0,
        float CV4_OM2_3,
        float CV5_OM2_3,
        float CV3_OM3_0,
        float CV4_OM3_0,
        float CV5_OM3_3,
        float CV4_OM4_0,
        float CV5_OM4_0,
        float CV5_OM5_0,
        float OM1_0_OM1_1,
        float OM1_1_OM1_2_E,
        float OM1_1_OM1_2_F,
        float OM1_1_OM1_3,
        float OM1_2_E_OM1_1,
        float OM1_2_F_OM1_1,
        float OM1_2_E_OM1_3,
        float OM1_2_F_OM1_3,
        float OM1_3_OM1_1,
        float OM1_3_OM1_2_E,
        float OM1_3_OM1_2_F,
        float OM1_2_E_eIP_E,
        float OM1_2_F_eIP_F,
        float OM1_2_F_OM2_2_F,
        float OM2_0_OM2_1,
        float OM2_1_OM2_2_E,
        float OM2_1_OM2_2_F,
        float OM2_1_OM2_3,
        float OM2_2_E_OM2_1,
        float OM2_2_F_OM2_1,
        float OM2_2_E_OM2_3,
        float OM2_2_F_OM2_3,
        float OM2_3_OM2_1,
        float OM2_3_OM2_2_E,
        float OM2_3_OM2_2_F,
        float OM2_2_E_eIP_E,
        float OM2_2_F_eIP_F,
        float OM2_2_F_OM3_2_F,
        float OM3_0_OM3_1,
        float OM3_1_OM3_2_E,
        float OM3_1_OM3_2_F,
        float OM3_1_OM3_3,
        float OM3_2_E_OM3_1,
        float OM3_2_F_OM3_1,
        float OM3_2_E_OM3_3,
        float OM3_2_F_OM3_3,
        float OM3_3_OM3_1,
        float OM3_3_OM3_2_E,
        float OM3_3_OM3_2_F,
        float OM3_2_E_eIP_E,
        float OM3_2_F_eIP_F,
        float OM3_2_F_OM4_2_F,
        float OM4_0_OM4_1,
        float OM4_1_OM4_2_E,
        float OM4_1_OM4_2_F,
        float OM4_1_OM4_3,
        float OM4_2_E_OM4_1,
        float OM4_2_F_OM4_1,
        float OM4_2_E_OM4_3,
        float OM4_2_F_OM4_3,
        float OM4_3_OM4_1,
        float OM4_3_OM4_2_E,
        float OM4_3_OM4_2_F,
        float OM4_2_E_eIP_E,
        float OM4_2_F_eIP_F,
        float OM4_2_F_OM5_2_F,
        float OM5_0_OM5_1,
        float OM5_1_OM5_2_E,
        float OM5_1_OM5_2_F,
        float OM5_1_OM5_3,
        float OM5_2_E_OM5_1,
        float OM5_2_F_OM5_1,
        float OM5_2_E_OM5_3,
        float OM5_2_F_OM5_3,
        float OM5_3_OM5_1,
        float OM5_3_OM5_2_E,
        float OM5_3_OM5_2_F,
        float OM5_2_E_eIP_E,
        float OM5_2_F_eIP_F,
        float CV1_iIP_E,
        float CV2_iIP_E,
        float CV3_iIP_E,
        float CV4_iIP_E,
        float CV5_iIP_E,
        float iIP_E_eIP_F,
        float iIP_F_eIP_E,
        float iIP_E_OM1_2_F,
        float iIP_E_OM2_2_F,
        float iIP_E_OM3_2_F,
        float iIP_E_OM4_2_F,
        float eIP_E_MN_E,
        float eIP_F_MN_F,
        float iIP_E_Ia_E_pool,
        float iIP_F_Ia_F_pool,
        float Ia_E_pool_MN_F,
        float Ia_E_pool_Ia_F_pool,
        float Ia_F_pool_MN_E,
        float Ia_F_pool_Ia_E_pool,
        float Ia_E_aff_MN_E,
        float Ia_F_aff_MN_F,
        float MN_E_R_E,
        float MN_F_R_F,
        float R_E_MN_E,
        float R_E_R_F,
        float R_F_MN_F,
        float R_F_R_E,
        float E1_OM1_0_d,
        float E2_OM2_0_d,
        float E3_OM3_0_d,
        float E4_OM4_0_d,
        float E5_OM5_0_d,
        float CV1_OM1_0_d,
        float CV2_OM1_0_d,
        float CV3_OM1_3_d,
        float CV4_OM1_3_d,
        float CV5_OM1_3_d,
        float CV2_OM2_0_d,
        float CV3_OM2_0_d,
        float CV4_OM2_3_d,
        float CV5_OM2_3_d,
        float CV3_OM3_0_d,
        float CV4_OM3_0_d,
        float CV5_OM3_3_d,
        float CV4_OM4_0_d,
        float CV5_OM4_0_d,
        float CV5_OM5_0_d,
        float OM1_0_OM1_1_d,
        float OM1_1_OM1_2_E_d,
        float OM1_1_OM1_2_F_d,
        float OM1_1_OM1_3_d,
        float OM1_2_E_OM1_1_d,
        float OM1_2_F_OM1_1_d,
        float OM1_2_E_OM1_3_d,
        float OM1_2_F_OM1_3_d,
        float OM1_3_OM1_1_d,
        float OM1_3_OM1_2_E_d,
        float OM1_3_OM1_2_F_d,
        float OM1_2_E_eIP_E_d,
        float OM1_2_F_eIP_F_d,
        float OM1_2_F_OM2_2_F_d,
        float OM2_0_OM2_1_d,
        float OM2_1_OM2_2_E_d,
        float OM2_1_OM2_2_F_d,
        float OM2_1_OM2_3_d,
        float OM2_2_E_OM2_1_d,
        float OM2_2_F_OM2_1_d,
        float OM2_2_E_OM2_3_d,
        float OM2_2_F_OM2_3_d,
        float OM2_3_OM2_1_d,
        float OM2_3_OM2_2_E_d,
        float OM2_3_OM2_2_F_d,
        float OM2_2_E_eIP_E_d,
        float OM2_2_F_eIP_F_d,
        float OM2_2_F_OM3_2_F_d,
        float OM3_0_OM3_1_d,
        float OM3_1_OM3_2_E_d,
        float OM3_1_OM3_2_F_d,
        float OM3_1_OM3_3_d,
        float OM3_2_E_OM3_1_d,
        float OM3_2_F_OM3_1_d,
        float OM3_2_E_OM3_3_d,
        float OM3_2_F_OM3_3_d,
        float OM3_3_OM3_1_d,
        float OM3_3_OM3_2_E_d,
        float OM3_3_OM3_2_F_d,
        float OM3_2_E_eIP_E_d,
        float OM3_2_F_eIP_F_d,
        float OM3_2_F_OM4_2_F_d,
        float OM4_0_OM4_1_d,
        float OM4_1_OM4_2_E_d,
        float OM4_1_OM4_2_F_d,
        float OM4_1_OM4_3_d,
        float OM4_2_E_OM4_1_d,
        float OM4_2_F_OM4_1_d,
        float OM4_2_E_OM4_3_d,
        float OM4_2_F_OM4_3_d,
        float OM4_3_OM4_1_d,
        float OM4_3_OM4_2_E_d,
        float OM4_3_OM4_2_F_d,
        float OM4_2_E_eIP_E_d,
        float OM4_2_F_eIP_F_d,
        float OM4_2_F_OM5_2_F_d,
        float OM5_0_OM5_1_d,
        float OM5_1_OM5_2_E_d,
        float OM5_1_OM5_2_F_d,
        float OM5_1_OM5_3_d,
        float OM5_2_E_OM5_1_d,
        float OM5_2_F_OM5_1_d,
        float OM5_2_E_OM5_3_d,
        float OM5_2_F_OM5_3_d,
        float OM5_3_OM5_1_d,
        float OM5_3_OM5_2_E_d,
        float OM5_3_OM5_2_F_d,
        float OM5_2_E_eIP_E_d,
        float OM5_2_F_eIP_F_d,
        float CV1_iIP_E_d,
        float CV2_iIP_E_d,
        float CV3_iIP_E_d,
        float CV4_iIP_E_d,
        float CV5_iIP_E_d,
        float iIP_E_eIP_F_d,
        float iIP_F_eIP_E_d,
        float iIP_E_OM1_2_F_d,
        float iIP_E_OM2_2_F_d,
        float iIP_E_OM3_2_F_d,
        float iIP_E_OM4_2_F_d,
        float eIP_E_MN_E_d,
        float eIP_F_MN_F_d,
        float iIP_E_Ia_E_pool_d,
        float iIP_F_Ia_F_pool_d,
        float Ia_E_pool_MN_F_d,
        float Ia_E_pool_Ia_F_pool_d,
        float Ia_F_pool_MN_E_d,
        float Ia_F_pool_Ia_E_pool_d,
        float Ia_E_aff_MN_E_d,
        float Ia_F_aff_MN_F_d,
        float MN_E_R_E_d,
        float MN_F_R_F_d,
        float R_E_MN_E_d,
        float R_E_R_F_d,
        float R_F_MN_F_d,
        float R_F_R_E_d) {

    float inh_coef = 1;

    /// groups of neurons
    Group EES = form_group("EES");
    Group E1 = form_group("E1");
    Group E2 = form_group("E2");
    Group E3 = form_group("E3");
    Group E4 = form_group("E4");
    Group E5 = form_group("E5");

    Group CV1 = form_group("CV1", 1);
    Group CV2 = form_group("CV2", 1);
    Group CV3 = form_group("CV3", 1);
    Group CV4 = form_group("CV4", 1);
    Group CV5 = form_group("CV5", 1);
    Group CD4 = form_group("CD4", 1);
    Group CD5 = form_group("CD5", 1);

    Group OM1_0 = form_group("OM1_0");
    Group OM1_1 = form_group("OM1_1");
    Group OM1_2_E = form_group("OM1_2_E");
    Group OM1_2_F = form_group("OM1_2_F");
    Group OM1_3 = form_group("OM1_3");

    Group OM2_0 = form_group("OM2_0");
    Group OM2_1 = form_group("OM2_1");
    Group OM2_2_E = form_group("OM2_2_E");
    Group OM2_2_F = form_group("OM2_2_F");
    Group OM2_3 = form_group("OM2_3");

    Group OM3_0 = form_group("OM3_0");
    Group OM3_1 = form_group("OM3_1");
    Group OM3_2_E = form_group("OM3_2_E");
    Group OM3_2_F = form_group("OM3_2_F");
    Group OM3_3 = form_group("OM3_3");

    Group OM4_0 = form_group("OM4_0");
    Group OM4_1 = form_group("OM4_1");
    Group OM4_2_E = form_group("OM4_2_E");
    Group OM4_2_F = form_group("OM4_2_F");
    Group OM4_3 = form_group("OM4_3");

    Group OM5_0 = form_group("OM5_0");
    Group OM5_1 = form_group("OM5_1");
    Group OM5_2_E = form_group("OM5_2_E");
    Group OM5_2_F = form_group("OM5_2_F");
    Group OM5_3 = form_group("OM5_3");

    Group MN_E = form_group("MN_E", neurons_in_moto_extensor);
    Group MN_F = form_group("MN_F", neurons_in_moto_flexor);

    Group Ia_E_aff = form_group("Ia_E_aff", neurons_in_afferent);
    Group Ia_F_aff = form_group("Ia_F_aff", neurons_in_afferent);

    Group R_E = form_group("R_E");
    Group R_F = form_group("R_F");

    Group Ia_E_pool = form_group("Ia_E_pool", neurons_in_aff_ip);
    Group Ia_F_pool = form_group("Ia_F_pool", neurons_in_aff_ip);

    Group eIP_E = form_group("eIP_E", neurons_in_ip);
    Group eIP_F = form_group("eIP_F", neurons_in_ip);

    Group iIP_E = form_group("iIP_E", neurons_in_ip);
    Group iIP_F = form_group("iIP_F", neurons_in_ip);


    /// connectomes
    connect_fixed_outdegree(EES, E1, 0.5, 0.5);
    connect_fixed_outdegree(E1, E2, 0.5, 0.2);
    connect_fixed_outdegree(E2, E3, 0.5, 0.2);
    connect_fixed_outdegree(E3, E4, 0.5, 0.2);
    connect_fixed_outdegree(E4, E5, 0.5, 0.2);

    /// OM 1
    // input from EES group 1
    connect_fixed_outdegree(E1, OM1_0, E1_OM1_0_d, E1_OM1_0);
    // input from sensory
    connect_one_to_all(CV1, OM1_0, CV1_OM1_0_d, CV1_OM1_0);
    connect_one_to_all(CV2, OM1_0, CV2_OM1_0_d, CV2_OM1_0);
    // [inhibition]
    connect_one_to_all(CV3, OM1_3, CV3_OM1_3_d, CV3_OM1_3);
    connect_one_to_all(CV4, OM1_3, CV4_OM1_3_d, CV4_OM1_3);
    connect_one_to_all(CV5, OM1_3, CV5_OM1_3_d, CV5_OM1_3);
    // inner connectomes
    connect_fixed_outdegree(OM1_0, OM1_1, OM1_0_OM1_1_d, OM1_0_OM1_1);
    connect_fixed_outdegree(OM1_1, OM1_2_E, OM1_1_OM1_2_F_d, OM1_1_OM1_2_F);
    connect_fixed_outdegree(OM1_1, OM1_2_F, OM1_1_OM1_2_F_d, OM1_1_OM1_2_F);
    connect_fixed_outdegree(OM1_1, OM1_3, OM1_1_OM1_3_d, OM1_1_OM1_3);
    connect_fixed_outdegree(OM1_2_E, OM1_1, OM1_2_E_OM1_1_d, OM1_2_E_OM1_1);
    connect_fixed_outdegree(OM1_2_F, OM1_1, OM1_2_F_OM1_1_d, OM1_2_F_OM1_1);
    connect_fixed_outdegree(OM1_2_E, OM1_3, OM1_2_E_OM1_3_d, OM1_2_E_OM1_3);
    connect_fixed_outdegree(OM1_2_F, OM1_3, OM1_2_F_OM1_3_d, OM1_2_F_OM1_3);
    connect_fixed_outdegree(OM1_3, OM1_1, OM1_3_OM1_1_d, -OM1_3_OM1_1 * inh_coef);
    connect_fixed_outdegree(OM1_3, OM1_2_E, OM1_3_OM1_2_E_d, -OM1_3_OM1_2_E * inh_coef);
    connect_fixed_outdegree(OM1_3, OM1_2_F, OM1_3_OM1_2_F_d, -OM1_3_OM1_2_F * inh_coef);
    // output to OM2
    connect_fixed_outdegree(OM1_2_F, OM2_2_F, OM1_2_F_OM2_2_F_d, OM1_2_F_OM2_2_F);
    // output to IP
    connect_fixed_outdegree(OM1_2_E, eIP_E, OM1_2_E_eIP_E_d, OM1_2_E_eIP_E, neurons_in_ip);
    connect_fixed_outdegree(OM1_2_F, eIP_F, OM1_2_F_eIP_F_d, OM1_2_F_eIP_F, neurons_in_ip);

    /// OM 2
    // input from EES group 2
    connect_fixed_outdegree(E2, OM2_0, E2_OM2_0_d, E2_OM2_0);
    // input from sensory [CV]
    connect_one_to_all(CV2, OM2_0, CV2_OM2_0_d, CV2_OM2_0);
    connect_one_to_all(CV3, OM2_0, CV3_OM2_0_d, CV3_OM2_0);
    // [inhibition]
    connect_one_to_all(CV4, OM2_3, CV4_OM2_3_d, CV4_OM2_3);
    connect_one_to_all(CV5, OM2_3, CV5_OM2_3_d, CV5_OM2_3);
    // inner connectomes
    connect_fixed_outdegree(OM2_0, OM2_1, OM2_0_OM2_1_d, OM2_0_OM2_1);
    connect_fixed_outdegree(OM2_1, OM2_2_E, OM2_1_OM2_2_F_d, OM2_1_OM2_2_F);
    connect_fixed_outdegree(OM2_1, OM2_2_F, OM2_1_OM2_2_F_d, OM2_1_OM2_2_F);
    connect_fixed_outdegree(OM2_1, OM2_3, OM2_1_OM2_3_d, OM2_1_OM2_3);
    connect_fixed_outdegree(OM2_2_E, OM2_1, OM2_2_E_OM2_1_d, OM2_2_E_OM2_1);
    connect_fixed_outdegree(OM2_2_F, OM2_1, OM2_2_F_OM2_1_d, OM2_2_F_OM2_1);
    connect_fixed_outdegree(OM2_2_E, OM2_3, OM2_2_E_OM2_3_d, OM2_2_E_OM2_3);
    connect_fixed_outdegree(OM2_2_F, OM2_3, OM2_2_F_OM2_3_d, OM2_2_F_OM2_3);
    connect_fixed_outdegree(OM2_3, OM2_1, OM2_3_OM2_1_d, -OM2_3_OM2_1 * inh_coef);
    connect_fixed_outdegree(OM2_3, OM2_2_E, OM2_3_OM2_2_E_d, -OM2_3_OM2_2_E * inh_coef);
    connect_fixed_outdegree(OM2_3, OM2_2_F, OM2_3_OM2_2_F_d, -OM2_3_OM2_2_F * inh_coef);
    // output to OM3
    connect_fixed_outdegree(OM2_2_F, OM3_2_F, OM2_2_F_OM3_2_F_d, OM2_2_F_OM3_2_F);
    // output to IP
    connect_fixed_outdegree(OM2_2_E, eIP_E, OM2_2_E_eIP_E_d, OM2_2_E_eIP_E, neurons_in_ip);
    connect_fixed_outdegree(OM2_2_F, eIP_F, OM2_2_F_eIP_F_d, OM2_2_F_eIP_F, neurons_in_ip);

    /// OM 3
    // input from EES group 3
    connect_fixed_outdegree(E3, OM3_0, E3_OM3_0_d, E3_OM3_0);
    // input from sensory [CV]
    connect_one_to_all(CV3, OM3_0, CV3_OM3_0_d, CV3_OM3_0);
    connect_one_to_all(CV4, OM3_0, CV4_OM3_0_d, CV4_OM3_0);
    // [inhibition]
    connect_one_to_all(CV5, OM3_3, CV5_OM3_3_d, CV5_OM3_3);
    // input from sensory [CD]
    connect_one_to_all(CD4, OM3_0, 1, 11);
    // inner connectomes
    connect_fixed_outdegree(OM3_0, OM3_1, OM3_0_OM3_1_d, OM3_0_OM3_1);
    connect_fixed_outdegree(OM3_1, OM3_2_E, OM3_1_OM3_2_F_d, OM3_1_OM3_2_F);
    connect_fixed_outdegree(OM3_1, OM3_2_F, OM3_1_OM3_2_F_d, OM3_1_OM3_2_F);
    connect_fixed_outdegree(OM3_1, OM3_3, OM3_1_OM3_3_d, OM3_1_OM3_3);
    connect_fixed_outdegree(OM3_2_E, OM3_1, OM3_2_E_OM3_1_d, OM3_2_E_OM3_1);
    connect_fixed_outdegree(OM3_2_F, OM3_1, OM3_2_F_OM3_1_d, OM3_2_F_OM3_1);
    connect_fixed_outdegree(OM3_2_E, OM3_3, OM3_2_E_OM3_3_d, OM3_2_E_OM3_3);
    connect_fixed_outdegree(OM3_2_F, OM3_3, OM3_2_F_OM3_3_d, OM3_2_F_OM3_3);
    connect_fixed_outdegree(OM3_3, OM3_1, OM3_3_OM3_1_d, -OM3_3_OM3_1 * inh_coef);
    connect_fixed_outdegree(OM3_3, OM3_2_E, OM3_3_OM3_2_E_d, -OM3_3_OM3_2_E * inh_coef);
    connect_fixed_outdegree(OM3_3, OM3_2_F, OM3_3_OM3_2_F_d, -OM3_3_OM3_2_F * inh_coef);
    // output to OM4
    connect_fixed_outdegree(OM3_2_F, OM4_2_F, OM3_2_F_OM4_2_F_d, OM3_2_F_OM4_2_F);
    // output to IP
    connect_fixed_outdegree(OM3_2_E, eIP_E, OM3_2_E_eIP_E_d, OM3_2_E_eIP_E, neurons_in_ip);
    connect_fixed_outdegree(OM3_2_F, eIP_F, OM3_2_F_eIP_F_d, OM3_2_F_eIP_F, neurons_in_ip);

    /// OM 4
    // input from EES group 4
    connect_fixed_outdegree(E4, OM4_0, E4_OM4_0_d, E4_OM4_0);
    // input from sensory [CV]
    connect_one_to_all(CV4, OM4_0, CV4_OM4_0_d, CV4_OM4_0);
    connect_one_to_all(CV5, OM4_0, CV5_OM4_0_d, CV5_OM4_0);
    // input from sensory [CD]
    connect_one_to_all(CD4, OM4_0, 1, 11);
    connect_one_to_all(CD5, OM4_0, 1, 11);
    // inner connectomes
    connect_fixed_outdegree(OM4_0, OM4_1, OM4_0_OM4_1_d, OM4_0_OM4_1);
    connect_fixed_outdegree(OM4_1, OM4_2_E, OM4_1_OM4_2_F_d, OM4_1_OM4_2_F);
    connect_fixed_outdegree(OM4_1, OM4_2_F, OM4_1_OM4_2_F_d, OM4_1_OM4_2_F);
    connect_fixed_outdegree(OM4_1, OM4_3, OM4_1_OM4_3_d, OM4_1_OM4_3);
    connect_fixed_outdegree(OM4_2_E, OM4_1, OM4_2_E_OM4_1_d, OM4_2_E_OM4_1);
    connect_fixed_outdegree(OM4_2_F, OM4_1, OM4_2_F_OM4_1_d, OM4_2_F_OM4_1);
    connect_fixed_outdegree(OM4_2_E, OM4_3, OM4_2_E_OM4_3_d, OM4_2_E_OM4_3);
    connect_fixed_outdegree(OM4_2_F, OM4_3, OM4_2_F_OM4_3_d, OM4_2_F_OM4_3);
    connect_fixed_outdegree(OM4_3, OM4_1, OM4_3_OM4_1_d, -OM4_3_OM4_1 * inh_coef);
    connect_fixed_outdegree(OM4_3, OM4_2_E, OM4_3_OM4_2_E_d, -OM4_3_OM4_2_E * inh_coef);
    connect_fixed_outdegree(OM4_3, OM4_2_F, OM4_3_OM4_2_F_d, -OM4_3_OM4_2_F * inh_coef);
    // output to OM5
    connect_fixed_outdegree(OM4_2_F, OM5_2_F, OM4_2_F_OM5_2_F_d, OM4_2_F_OM5_2_F);
    // output to IP
    connect_fixed_outdegree(OM4_2_E, eIP_E, OM1_2_E_eIP_E_d, OM1_2_E_eIP_E, neurons_in_ip);
    connect_fixed_outdegree(OM4_2_F, eIP_F, OM1_2_F_eIP_F_d, OM1_2_F_eIP_F, neurons_in_ip);

    /// OM 5
    // input from EES group 5
    connect_fixed_outdegree(E5, OM5_0, E5_OM5_0_d, E5_OM5_0);
    // input from sensory [CV]
    connect_one_to_all(CV5, OM5_0, CV5_OM5_0_d, CV5_OM5_0);
    // input from sensory [CD]
    connect_one_to_all(CD5, OM5_0, 1, 11);
    // inner connectomes
    connect_fixed_outdegree(OM5_0, OM5_1, OM5_0_OM5_1_d, OM5_0_OM5_1);
    connect_fixed_outdegree(OM5_1, OM5_2_E, OM5_1_OM5_2_F_d, OM5_1_OM5_2_F);
    connect_fixed_outdegree(OM5_1, OM5_2_F, OM5_1_OM5_2_F_d, OM5_1_OM5_2_F);
    connect_fixed_outdegree(OM5_1, OM5_3, OM5_1_OM5_3_d, OM5_1_OM5_3);
    connect_fixed_outdegree(OM5_2_E, OM5_1, OM5_2_E_OM5_1_d, OM5_2_E_OM5_1);
    connect_fixed_outdegree(OM5_2_F, OM5_1, OM5_2_F_OM5_1_d, OM5_2_F_OM5_1);
    connect_fixed_outdegree(OM5_2_E, OM5_3, OM5_2_E_OM5_3_d, OM5_2_E_OM5_3);
    connect_fixed_outdegree(OM5_2_F, OM5_3, OM5_2_F_OM5_3_d, OM5_2_F_OM5_3);
    connect_fixed_outdegree(OM5_3, OM5_1, OM5_3_OM5_1_d, -OM5_3_OM5_1 * inh_coef);
    connect_fixed_outdegree(OM5_3, OM5_2_E, OM5_3_OM5_2_E_d, -OM5_3_OM5_2_E * inh_coef);
    connect_fixed_outdegree(OM5_3, OM5_2_F, OM5_3_OM5_2_F_d, -OM5_3_OM5_2_F * inh_coef);
    // output to IP
    connect_fixed_outdegree(OM5_2_E, eIP_E, OM5_2_E_eIP_E_d, OM5_2_E_eIP_E, neurons_in_ip);
    connect_fixed_outdegree(OM5_2_F, eIP_F, OM5_2_F_eIP_F_d, OM5_2_F_eIP_F, neurons_in_ip);

    /// reflex arc
    connect_fixed_outdegree(EES, Ia_E_aff, 1, 500);
    connect_fixed_outdegree(EES, Ia_F_aff, 1, 500);

    connect_one_to_all(CV1, iIP_E, CV1_iIP_E_d, CV1_iIP_E);
    connect_one_to_all(CV2, iIP_E, CV2_iIP_E_d, CV2_iIP_E);
    connect_one_to_all(CV3, iIP_E, CV3_iIP_E_d, CV3_iIP_E);
    connect_one_to_all(CV4, iIP_E, CV4_iIP_E_d, CV4_iIP_E);
    connect_one_to_all(CV5, iIP_E, CV5_iIP_E_d, CV5_iIP_E);

    connect_fixed_outdegree(iIP_E, eIP_F, iIP_E_eIP_F_d, iIP_E_eIP_F);
    connect_fixed_outdegree(iIP_F, eIP_E, iIP_F_eIP_E_d, iIP_F_eIP_E);

    connect_fixed_outdegree(iIP_E, OM1_2_F, iIP_E_OM1_2_F_d, -iIP_E_OM1_2_F);
    connect_fixed_outdegree(iIP_E, OM2_2_F, iIP_E_OM2_2_F_d, -iIP_E_OM2_2_F);
    connect_fixed_outdegree(iIP_E, OM3_2_F, iIP_E_OM3_2_F_d, -iIP_E_OM3_2_F);
    connect_fixed_outdegree(iIP_E, OM4_2_F, iIP_E_OM4_2_F_d, -iIP_E_OM4_2_F);

    connect_fixed_outdegree(eIP_E, MN_E, eIP_E_MN_E_d, eIP_E_MN_E);
    connect_fixed_outdegree(eIP_F, MN_F, eIP_F_MN_F_d, eIP_F_MN_F);

    connect_fixed_outdegree(iIP_E, Ia_E_pool, iIP_E_Ia_E_pool_d, iIP_E_Ia_E_pool);
    connect_fixed_outdegree(iIP_F, Ia_F_pool, iIP_F_Ia_F_pool_d, iIP_F_Ia_F_pool);

    connect_fixed_outdegree(Ia_E_pool, MN_F, Ia_E_pool_MN_F_d, -Ia_E_pool_MN_F);
    connect_fixed_outdegree(Ia_E_pool, Ia_F_pool, Ia_E_pool_Ia_F_pool_d, -Ia_E_pool_Ia_F_pool);
    connect_fixed_outdegree(Ia_F_pool, MN_E, Ia_F_pool_MN_E_d, -Ia_F_pool_MN_E);
    connect_fixed_outdegree(Ia_F_pool, Ia_E_pool, Ia_F_pool_Ia_E_pool_d, -Ia_F_pool_Ia_E_pool);

    connect_fixed_outdegree(Ia_E_aff, MN_E, Ia_E_aff_MN_E_d, Ia_E_aff_MN_E);
    connect_fixed_outdegree(Ia_F_aff, MN_F, Ia_F_aff_MN_F_d, Ia_F_aff_MN_F);

    connect_fixed_outdegree(MN_E, R_E, MN_E_R_E_d, MN_E_R_E);
    connect_fixed_outdegree(MN_F, R_F, MN_F_R_F_d, MN_F_R_F);

    connect_fixed_outdegree(R_E, MN_E, R_E_MN_E_d, -R_E_MN_E);
    connect_fixed_outdegree(R_E, R_F, R_E_R_F_d, -R_E_R_F);

    connect_fixed_outdegree(R_F, MN_F, R_F_MN_F_d, -R_F_MN_F);
    connect_fixed_outdegree(R_F, R_E, R_F_R_E_d, -R_F_R_E);

}


void save(int test_index, GroupMetadata &metadata, const string& folder){
    /**
     *
     */
    ofstream file;
    string file_name = "/" + metadata.group.group_name + ".dat";

    file.open(folder + file_name);
    // save voltage
    for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
        file << metadata.voltage_array[sim_iter] << " ";
    file << endl;

    // save g_exc
    for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
        file << metadata.g_exc[sim_iter] << " ";
    file << endl;

    // save g_inh
    for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
        file << metadata.g_inh[sim_iter] << " ";
    file << endl;

    // save spikes
    for (float const& value: metadata.spike_vector) {
        file << value << " ";
    }
    file.close();

    cout << "Saved to: " << folder + file_name << endl;
}

void save_result(int test_index, int save_all) {
    /**
     *
     */
    string current_path = getcwd(nullptr, 0);

    printf("[Test #%d] Save %s results to: %s \n", test_index, (save_all == 0)? "MOTO" : "ALL", current_path.c_str());

    for(GroupMetadata &metadata : all_groups) {
        if (save_all == 0) {
            if(metadata.group.group_name == "MN_E")
                save(test_index, metadata, current_path);
            if(metadata.group.group_name == "MN_F")
                save(test_index, metadata, current_path);
        } else {
            save(test_index, metadata, current_path);
        }
    }
}

// copy data from host to device
template <typename type>
void memcpyHtD(type* gpu, type* host, unsigned int size) {
    cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

// copy data from device to host
template <typename type>
void memcpyDtH(type* host, type* gpu, unsigned int size) {
    cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

// get datasize of current variable type and its number
template <typename type>
unsigned int datasize(unsigned int size) {
    return sizeof(type) * size;
}

// fill array with current value
template <typename type>
void init_array(type *array, unsigned int size, type value) {
    for(int i = 0; i < size; i++)
        array[i] = value;
}
// fill array by normal distribution
template <typename type>
void rand_normal_init_array(type *array, unsigned int size, type mean, type stddev) {
    random_device r;
    default_random_engine generator(r());
    normal_distribution<float> distr(mean, stddev);

    for(unsigned int i = 0; i < size; i++)
        array[i] = (type)distr(generator);
}

int get_skin_stim_time(int cms) {
    if (cms == 21)
        return 25;
    if (cms == 15)
        return 50;
    return 125;
}

void copy_data_to(GroupMetadata &metadata,
                  const float* nrn_v_m,
                  const float* nrn_g_exc,
                  const float* nrn_g_inh,
                  const bool *nrn_has_spike,
                  unsigned int sim_iter) {
    /**
     *
     */
    float nrn_mean_volt = 0;
    float nrn_mean_g_exc = 0;
    float nrn_mean_g_inh = 0;

    for(unsigned int tid = metadata.group.id_start; tid <= metadata.group.id_end; tid++) {
        nrn_mean_volt += nrn_v_m[tid];
        nrn_mean_g_exc += nrn_g_exc[tid];
        nrn_mean_g_inh += nrn_g_inh[tid];
        if (nrn_has_spike[tid]) {
            metadata.spike_vector.push_back(step_to_ms(sim_iter) + 0.25);
        }
    }
    metadata.g_exc[sim_iter] = nrn_mean_g_exc / metadata.group.group_size;
    metadata.g_inh[sim_iter] = nrn_mean_g_inh / metadata.group.group_size;
    metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size;
}

__host__
void bimodal_distr_for_moto_neurons(float *nrn_diameter) {

    int loc_standby = 57;
    int scale_standby = 6;
    int loc_active = 27;
    int scale_active = 3;

    int MN_E_start = 1557;
    int MN_E_end = 1766;
    int MN_F_start = 1767;
    int MN_F_end = 1946;

    int nrn_number_extensor = MN_E_end - MN_E_start;
    int nrn_number_flexor = MN_F_end - MN_F_start;

    int standby_percent = 70;

    int standby_size_extensor = (int)(nrn_number_extensor * standby_percent / 100);
    int standby_size_flexor = (int)(nrn_number_flexor * standby_percent / 100);

    random_device r;
    default_random_engine generator(r());

    normal_distribution<float> g_standby(loc_standby, scale_standby);
    normal_distribution<float> g_active(loc_active, scale_active);

    /***
     * fill 70% neurons with diameters (57, 6)
     * 30% neurons with diameters (27, 3)
     ***/

    for (int i = MN_E_start; i < MN_E_start + standby_size_extensor; i++) {
        nrn_diameter[i] = g_standby(generator);
    }
    for (int i = MN_E_start + standby_size_extensor; i <= MN_E_end; i++) {
        nrn_diameter[i] = g_active(generator);
    }

    for (int i = MN_F_start; i < MN_F_start + standby_size_flexor; i++) {
        nrn_diameter[i] = g_standby(generator);
    }
    for (int i = MN_F_start + standby_size_flexor; i <= MN_F_end; i++) {
        nrn_diameter[i] = g_active(generator);
    }

//    diameters = np.concatenate([np.random.normal(loc=loc_active, scale=scale_active, size=active_size),
//            np.random.normal(loc=loc_standby, scale=scale_standby, size=standby_size)])

}



__host__
void simulate(int cms, int ees, int inh, int ped, int ht5, int save_all, int itest,
              float E1_OM1_0,
              float E2_OM2_0,
              float E3_OM3_0,
              float E4_OM4_0,
              float E5_OM5_0,
              float CV1_OM1_0,
              float CV2_OM1_0,
              float CV3_OM1_3,
              float CV4_OM1_3,
              float CV5_OM1_3,
              float CV2_OM2_0,
              float CV3_OM2_0,
              float CV4_OM2_3,
              float CV5_OM2_3,
              float CV3_OM3_0,
              float CV4_OM3_0,
              float CV5_OM3_3,
              float CV4_OM4_0,
              float CV5_OM4_0,
              float CV5_OM5_0,
              float OM1_0_OM1_1,
              float OM1_1_OM1_2_E,
              float OM1_1_OM1_2_F,
              float OM1_1_OM1_3,
              float OM1_2_E_OM1_1,
              float OM1_2_F_OM1_1,
              float OM1_2_E_OM1_3,
              float OM1_2_F_OM1_3,
              float OM1_3_OM1_1,
              float OM1_3_OM1_2_E,
              float OM1_3_OM1_2_F,
              float OM1_2_E_eIP_E,
              float OM1_2_F_eIP_F,
              float OM1_2_F_OM2_2_F,
              float OM2_0_OM2_1,
              float OM2_1_OM2_2_E,
              float OM2_1_OM2_2_F,
              float OM2_1_OM2_3,
              float OM2_2_E_OM2_1,
              float OM2_2_F_OM2_1,
              float OM2_2_E_OM2_3,
              float OM2_2_F_OM2_3,
              float OM2_3_OM2_1,
              float OM2_3_OM2_2_E,
              float OM2_3_OM2_2_F,
              float OM2_2_E_eIP_E,
              float OM2_2_F_eIP_F,
              float OM2_2_F_OM3_2_F,
              float OM3_0_OM3_1,
              float OM3_1_OM3_2_E,
              float OM3_1_OM3_2_F,
              float OM3_1_OM3_3,
              float OM3_2_E_OM3_1,
              float OM3_2_F_OM3_1,
              float OM3_2_E_OM3_3,
              float OM3_2_F_OM3_3,
              float OM3_3_OM3_1,
              float OM3_3_OM3_2_E,
              float OM3_3_OM3_2_F,
              float OM3_2_E_eIP_E,
              float OM3_2_F_eIP_F,
              float OM3_2_F_OM4_2_F,
              float OM4_0_OM4_1,
              float OM4_1_OM4_2_E,
              float OM4_1_OM4_2_F,
              float OM4_1_OM4_3,
              float OM4_2_E_OM4_1,
              float OM4_2_F_OM4_1,
              float OM4_2_E_OM4_3,
              float OM4_2_F_OM4_3,
              float OM4_3_OM4_1,
              float OM4_3_OM4_2_E,
              float OM4_3_OM4_2_F,
              float OM4_2_E_eIP_E,
              float OM4_2_F_eIP_F,
              float OM4_2_F_OM5_2_F,
              float OM5_0_OM5_1,
              float OM5_1_OM5_2_E,
              float OM5_1_OM5_2_F,
              float OM5_1_OM5_3,
              float OM5_2_E_OM5_1,
              float OM5_2_F_OM5_1,
              float OM5_2_E_OM5_3,
              float OM5_2_F_OM5_3,
              float OM5_3_OM5_1,
              float OM5_3_OM5_2_E,
              float OM5_3_OM5_2_F,
              float OM5_2_E_eIP_E,
              float OM5_2_F_eIP_F,
              float CV1_iIP_E,
              float CV2_iIP_E,
              float CV3_iIP_E,
              float CV4_iIP_E,
              float CV5_iIP_E,
              float iIP_E_eIP_F,
              float iIP_F_eIP_E,
              float iIP_E_OM1_2_F,
              float iIP_E_OM2_2_F,
              float iIP_E_OM3_2_F,
              float iIP_E_OM4_2_F,
              float eIP_E_MN_E,
              float eIP_F_MN_F,
              float iIP_E_Ia_E_pool,
              float iIP_F_Ia_F_pool,
              float Ia_E_pool_MN_F,
              float Ia_E_pool_Ia_F_pool,
              float Ia_F_pool_MN_E,
              float Ia_F_pool_Ia_E_pool,
              float Ia_E_aff_MN_E,
              float Ia_F_aff_MN_F,
              float MN_E_R_E,
              float MN_F_R_F,
              float R_E_MN_E,
              float R_E_R_F,
              float R_F_MN_F,
              float R_F_R_E,
              float E1_OM1_0_d,
              float E2_OM2_0_d,
              float E3_OM3_0_d,
              float E4_OM4_0_d,
              float E5_OM5_0_d,
              float CV1_OM1_0_d,
              float CV2_OM1_0_d,
              float CV3_OM1_3_d,
              float CV4_OM1_3_d,
              float CV5_OM1_3_d,
              float CV2_OM2_0_d,
              float CV3_OM2_0_d,
              float CV4_OM2_3_d,
              float CV5_OM2_3_d,
              float CV3_OM3_0_d,
              float CV4_OM3_0_d,
              float CV5_OM3_3_d,
              float CV4_OM4_0_d,
              float CV5_OM4_0_d,
              float CV5_OM5_0_d,
              float OM1_0_OM1_1_d,
              float OM1_1_OM1_2_E_d,
              float OM1_1_OM1_2_F_d,
              float OM1_1_OM1_3_d,
              float OM1_2_E_OM1_1_d,
              float OM1_2_F_OM1_1_d,
              float OM1_2_E_OM1_3_d,
              float OM1_2_F_OM1_3_d,
              float OM1_3_OM1_1_d,
              float OM1_3_OM1_2_E_d,
              float OM1_3_OM1_2_F_d,
              float OM1_2_E_eIP_E_d,
              float OM1_2_F_eIP_F_d,
              float OM1_2_F_OM2_2_F_d,
              float OM2_0_OM2_1_d,
              float OM2_1_OM2_2_E_d,
              float OM2_1_OM2_2_F_d,
              float OM2_1_OM2_3_d,
              float OM2_2_E_OM2_1_d,
              float OM2_2_F_OM2_1_d,
              float OM2_2_E_OM2_3_d,
              float OM2_2_F_OM2_3_d,
              float OM2_3_OM2_1_d,
              float OM2_3_OM2_2_E_d,
              float OM2_3_OM2_2_F_d,
              float OM2_2_E_eIP_E_d,
              float OM2_2_F_eIP_F_d,
              float OM2_2_F_OM3_2_F_d,
              float OM3_0_OM3_1_d,
              float OM3_1_OM3_2_E_d,
              float OM3_1_OM3_2_F_d,
              float OM3_1_OM3_3_d,
              float OM3_2_E_OM3_1_d,
              float OM3_2_F_OM3_1_d,
              float OM3_2_E_OM3_3_d,
              float OM3_2_F_OM3_3_d,
              float OM3_3_OM3_1_d,
              float OM3_3_OM3_2_E_d,
              float OM3_3_OM3_2_F_d,
              float OM3_2_E_eIP_E_d,
              float OM3_2_F_eIP_F_d,
              float OM3_2_F_OM4_2_F_d,
              float OM4_0_OM4_1_d,
              float OM4_1_OM4_2_E_d,
              float OM4_1_OM4_2_F_d,
              float OM4_1_OM4_3_d,
              float OM4_2_E_OM4_1_d,
              float OM4_2_F_OM4_1_d,
              float OM4_2_E_OM4_3_d,
              float OM4_2_F_OM4_3_d,
              float OM4_3_OM4_1_d,
              float OM4_3_OM4_2_E_d,
              float OM4_3_OM4_2_F_d,
              float OM4_2_E_eIP_E_d,
              float OM4_2_F_eIP_F_d,
              float OM4_2_F_OM5_2_F_d,
              float OM5_0_OM5_1_d,
              float OM5_1_OM5_2_E_d,
              float OM5_1_OM5_2_F_d,
              float OM5_1_OM5_3_d,
              float OM5_2_E_OM5_1_d,
              float OM5_2_F_OM5_1_d,
              float OM5_2_E_OM5_3_d,
              float OM5_2_F_OM5_3_d,
              float OM5_3_OM5_1_d,
              float OM5_3_OM5_2_E_d,
              float OM5_3_OM5_2_F_d,
              float OM5_2_E_eIP_E_d,
              float OM5_2_F_eIP_F_d,
              float CV1_iIP_E_d,
              float CV2_iIP_E_d,
              float CV3_iIP_E_d,
              float CV4_iIP_E_d,
              float CV5_iIP_E_d,
              float iIP_E_eIP_F_d,
              float iIP_F_eIP_E_d,
              float iIP_E_OM1_2_F_d,
              float iIP_E_OM2_2_F_d,
              float iIP_E_OM3_2_F_d,
              float iIP_E_OM4_2_F_d,
              float eIP_E_MN_E_d,
              float eIP_F_MN_F_d,
              float iIP_E_Ia_E_pool_d,
              float iIP_F_Ia_F_pool_d,
              float Ia_E_pool_MN_F_d,
              float Ia_E_pool_Ia_F_pool_d,
              float Ia_F_pool_MN_E_d,
              float Ia_F_pool_Ia_E_pool_d,
              float Ia_E_aff_MN_E_d,
              float Ia_F_aff_MN_F_d,
              float MN_E_R_E_d,
              float MN_F_R_F_d,
              float R_E_MN_E_d,
              float R_E_R_F_d,
              float R_F_MN_F_d,
              float R_F_R_E_d) {
	/**
	 *
	 */
	chrono::time_point<chrono::system_clock> simulation_t_start, simulation_t_end;

	const unsigned int skin_stim_time = get_skin_stim_time(cms);
	const unsigned int T_simulation = 11 * skin_stim_time * LEG_STEPS;
	// calculate how much steps in simulation time [steps]
	SIM_TIME_IN_STEPS = (unsigned int)(T_simulation / SIM_STEP);

	// calculate spike frequency and C0/C1 activation time in steps
	auto ees_spike_each_step = (unsigned int)(1000 / ees / SIM_STEP);
	auto steps_activation_C0 = (unsigned int)(5 * get_skin_stim_time(25) / SIM_STEP);
//    auto steps_activation_C0 = (unsigned int)(5 * skin_stim_time / SIM_STEP);
    auto steps_activation_C1 = (unsigned int)(6 * skin_stim_time / SIM_STEP);
//    auto steps_activation_C1 = (unsigned int)(6 * get_skin_stim_time(25) / SIM_STEP);

	// init neuron groups and connectomes
    init_network(E1_OM1_0,
                 E2_OM2_0,
                 E3_OM3_0,
                 E4_OM4_0,
                 E5_OM5_0,
                 CV1_OM1_0,
                 CV2_OM1_0,
                 CV3_OM1_3,
                 CV4_OM1_3,
                 CV5_OM1_3,
                 CV2_OM2_0,
                 CV3_OM2_0,
                 CV4_OM2_3,
                 CV5_OM2_3,
                 CV3_OM3_0,
                 CV4_OM3_0,
                 CV5_OM3_3,
                 CV4_OM4_0,
                 CV5_OM4_0,
                 CV5_OM5_0,
                 OM1_0_OM1_1,
                 OM1_1_OM1_2_E,
                 OM1_1_OM1_2_F,
                 OM1_1_OM1_3,
                 OM1_2_E_OM1_1,
                 OM1_2_F_OM1_1,
                 OM1_2_E_OM1_3,
                 OM1_2_F_OM1_3,
                 OM1_3_OM1_1,
                 OM1_3_OM1_2_E,
                 OM1_3_OM1_2_F,
                 OM1_2_E_eIP_E,
                 OM1_2_F_eIP_F,
                 OM1_2_F_OM2_2_F,
                 OM2_0_OM2_1,
                 OM2_1_OM2_2_E,
                 OM2_1_OM2_2_F,
                 OM2_1_OM2_3,
                 OM2_2_E_OM2_1,
                 OM2_2_F_OM2_1,
                 OM2_2_E_OM2_3,
                 OM2_2_F_OM2_3,
                 OM2_3_OM2_1,
                 OM2_3_OM2_2_E,
                 OM2_3_OM2_2_F,
                 OM2_2_E_eIP_E,
                 OM2_2_F_eIP_F,
                 OM2_2_F_OM3_2_F,
                 OM3_0_OM3_1,
                 OM3_1_OM3_2_E,
                 OM3_1_OM3_2_F,
                 OM3_1_OM3_3,
                 OM3_2_E_OM3_1,
                 OM3_2_F_OM3_1,
                 OM3_2_E_OM3_3,
                 OM3_2_F_OM3_3,
                 OM3_3_OM3_1,
                 OM3_3_OM3_2_E,
                 OM3_3_OM3_2_F,
                 OM3_2_E_eIP_E,
                 OM3_2_F_eIP_F,
                 OM3_2_F_OM4_2_F,
                 OM4_0_OM4_1,
                 OM4_1_OM4_2_E,
                 OM4_1_OM4_2_F,
                 OM4_1_OM4_3,
                 OM4_2_E_OM4_1,
                 OM4_2_F_OM4_1,
                 OM4_2_E_OM4_3,
                 OM4_2_F_OM4_3,
                 OM4_3_OM4_1,
                 OM4_3_OM4_2_E,
                 OM4_3_OM4_2_F,
                 OM4_2_E_eIP_E,
                 OM4_2_F_eIP_F,
                 OM4_2_F_OM5_2_F,
                 OM5_0_OM5_1,
                 OM5_1_OM5_2_E,
                 OM5_1_OM5_2_F,
                 OM5_1_OM5_3,
                 OM5_2_E_OM5_1,
                 OM5_2_F_OM5_1,
                 OM5_2_E_OM5_3,
                 OM5_2_F_OM5_3,
                 OM5_3_OM5_1,
                 OM5_3_OM5_2_E,
                 OM5_3_OM5_2_F,
                 OM5_2_E_eIP_E,
                 OM5_2_F_eIP_F,
                 CV1_iIP_E,
                 CV2_iIP_E,
                 CV3_iIP_E,
                 CV4_iIP_E,
                 CV5_iIP_E,
                 iIP_E_eIP_F,
                 iIP_F_eIP_E,
                 iIP_E_OM1_2_F,
                 iIP_E_OM2_2_F,
                 iIP_E_OM3_2_F,
                 iIP_E_OM4_2_F,
                 eIP_E_MN_E,
                 eIP_F_MN_F,
                 iIP_E_Ia_E_pool,
                 iIP_F_Ia_F_pool,
                 Ia_E_pool_MN_F,
                 Ia_E_pool_Ia_F_pool,
                 Ia_F_pool_MN_E,
                 Ia_F_pool_Ia_E_pool,
                 Ia_E_aff_MN_E,
                 Ia_F_aff_MN_F,
                 MN_E_R_E,
                 MN_F_R_F,
                 R_E_MN_E,
                 R_E_R_F,
                 R_F_MN_F,
                 R_F_R_E,
                 E1_OM1_0_d,
                 E2_OM2_0_d,
                 E3_OM3_0_d,
                 E4_OM4_0_d,
                 E5_OM5_0_d,
                 CV1_OM1_0_d,
                 CV2_OM1_0_d,
                 CV3_OM1_3_d,
                 CV4_OM1_3_d,
                 CV5_OM1_3_d,
                 CV2_OM2_0_d,
                 CV3_OM2_0_d,
                 CV4_OM2_3_d,
                 CV5_OM2_3_d,
                 CV3_OM3_0_d,
                 CV4_OM3_0_d,
                 CV5_OM3_3_d,
                 CV4_OM4_0_d,
                 CV5_OM4_0_d,
                 CV5_OM5_0_d,
                 OM1_0_OM1_1_d,
                 OM1_1_OM1_2_E_d,
                 OM1_1_OM1_2_F_d,
                 OM1_1_OM1_3_d,
                 OM1_2_E_OM1_1_d,
                 OM1_2_F_OM1_1_d,
                 OM1_2_E_OM1_3_d,
                 OM1_2_F_OM1_3_d,
                 OM1_3_OM1_1_d,
                 OM1_3_OM1_2_E_d,
                 OM1_3_OM1_2_F_d,
                 OM1_2_E_eIP_E_d,
                 OM1_2_F_eIP_F_d,
                 OM1_2_F_OM2_2_F_d,
                 OM2_0_OM2_1_d,
                 OM2_1_OM2_2_E_d,
                 OM2_1_OM2_2_F_d,
                 OM2_1_OM2_3_d,
                 OM2_2_E_OM2_1_d,
                 OM2_2_F_OM2_1_d,
                 OM2_2_E_OM2_3_d,
                 OM2_2_F_OM2_3_d,
                 OM2_3_OM2_1_d,
                 OM2_3_OM2_2_E_d,
                 OM2_3_OM2_2_F_d,
                 OM2_2_E_eIP_E_d,
                 OM2_2_F_eIP_F_d,
                 OM2_2_F_OM3_2_F_d,
                 OM3_0_OM3_1_d,
                 OM3_1_OM3_2_E_d,
                 OM3_1_OM3_2_F_d,
                 OM3_1_OM3_3_d,
                 OM3_2_E_OM3_1_d,
                 OM3_2_F_OM3_1_d,
                 OM3_2_E_OM3_3_d,
                 OM3_2_F_OM3_3_d,
                 OM3_3_OM3_1_d,
                 OM3_3_OM3_2_E_d,
                 OM3_3_OM3_2_F_d,
                 OM3_2_E_eIP_E_d,
                 OM3_2_F_eIP_F_d,
                 OM3_2_F_OM4_2_F_d,
                 OM4_0_OM4_1_d,
                 OM4_1_OM4_2_E_d,
                 OM4_1_OM4_2_F_d,
                 OM4_1_OM4_3_d,
                 OM4_2_E_OM4_1_d,
                 OM4_2_F_OM4_1_d,
                 OM4_2_E_OM4_3_d,
                 OM4_2_F_OM4_3_d,
                 OM4_3_OM4_1_d,
                 OM4_3_OM4_2_E_d,
                 OM4_3_OM4_2_F_d,
                 OM4_2_E_eIP_E_d,
                 OM4_2_F_eIP_F_d,
                 OM4_2_F_OM5_2_F_d,
                 OM5_0_OM5_1_d,
                 OM5_1_OM5_2_E_d,
                 OM5_1_OM5_2_F_d,
                 OM5_1_OM5_3_d,
                 OM5_2_E_OM5_1_d,
                 OM5_2_F_OM5_1_d,
                 OM5_2_E_OM5_3_d,
                 OM5_2_F_OM5_3_d,
                 OM5_3_OM5_1_d,
                 OM5_3_OM5_2_E_d,
                 OM5_3_OM5_2_F_d,
                 OM5_2_E_eIP_E_d,
                 OM5_2_F_eIP_F_d,
                 CV1_iIP_E_d,
                 CV2_iIP_E_d,
                 CV3_iIP_E_d,
                 CV4_iIP_E_d,
                 CV5_iIP_E_d,
                 iIP_E_eIP_F_d,
                 iIP_F_eIP_E_d,
                 iIP_E_OM1_2_F_d,
                 iIP_E_OM2_2_F_d,
                 iIP_E_OM3_2_F_d,
                 iIP_E_OM4_2_F_d,
                 eIP_E_MN_E_d,
                 eIP_F_MN_F_d,
                 iIP_E_Ia_E_pool_d,
                 iIP_F_Ia_F_pool_d,
                 Ia_E_pool_MN_F_d,
                 Ia_E_pool_Ia_F_pool_d,
                 Ia_F_pool_MN_E_d,
                 Ia_F_pool_Ia_E_pool_d,
                 Ia_E_aff_MN_E_d,
                 Ia_F_aff_MN_F_d,
                 MN_E_R_E_d,
                 MN_F_R_F_d,
                 R_E_MN_E_d,
                 R_E_R_F_d,
                 R_F_MN_F_d,
                 R_F_R_E_d);

	const auto neurons_number = global_id;
	const unsigned int synapses_number = static_cast<int>(all_synapses.size());

	/// CPU variables
	// neuron variables
	float nrn_n[neurons_number];             // dimensionless quantity [0 .. 1] of potassium channel activation
	float nrn_h[neurons_number];             // dimensionless quantity [0 .. 1] of sodium channel activation
	float nrn_m[neurons_number];             // dimensionless quantity [0 .. 1] of sodium channel inactivation
	float nrn_v_m[neurons_number];           // [mV] neuron membrane potential
	float nrn_c_m[neurons_number];           // [mV] neuron membrane potential
	float nrn_g_exc[neurons_number];         // [nS] excitatory synapse exponential conductance
	float nrn_g_inh[neurons_number];         // [nS] inhibitory synapse exponential conductance
	float nrn_threshold[neurons_number];     // [mV] threshold levels
	bool nrn_has_spike[neurons_number];      // neuron state - has spike or not
	int nrn_ref_time[neurons_number];        // [step] neuron refractory time
	int nrn_ref_time_timer[neurons_number];  // [step] neuron refractory time timer
	float nrn_diameter[neurons_number];
	float nrn_g_Na[neurons_number];
	float nrn_g_K[neurons_number];
	float nrn_g_L[neurons_number];

	int len = 25;

	int begin_C_spiking[5] = {ms_to_step(0),
	                          ms_to_step(skin_stim_time),
	                          ms_to_step(2 * skin_stim_time),
	                          ms_to_step(3 * skin_stim_time),
	                          ms_to_step(5 * skin_stim_time)};

	int end_C_spiking[5] = {ms_to_step(skin_stim_time - 0.1),
	                        ms_to_step(2 * skin_stim_time - 0.1),
	                        ms_to_step(3 * skin_stim_time - 0.1),
	                        ms_to_step(5 * skin_stim_time - 0.1),
	                        ms_to_step(6 * skin_stim_time - 0.1)};

    float CV1_spikes_ms[len];
    float CV2_spikes_ms[len];
    float CV3_spikes_ms[len];
    float CV4_spikes_ms[len * 2];
    float CV5_spikes_ms[len];

    // fill arrays by initial data
	init_array<float>(nrn_n, neurons_number, 0);             // by default neurons have closed potassium channel
	init_array<float>(nrn_h, neurons_number, 1);             // by default neurons have opened sodium channel activation
	init_array<float>(nrn_m, neurons_number, 0);             // by default neurons have closed sodium channel inactivation
	init_array<float>(nrn_v_m, neurons_number, E_L);         // by default neurons have E_L membrane state at start
	init_array<float>(nrn_g_exc, neurons_number, 0);
	init_array<float>(nrn_g_inh, neurons_number, 0);         // by default neurons have zero inhibitory synaptic conductivity
	init_array<bool>(nrn_has_spike, neurons_number, false);  // by default neurons haven't spikes at start
	init_array<int>(nrn_ref_time_timer, neurons_number, 0);  // by default neurons have ref_t timers as 0
	init_array<float>(nrn_diameter, neurons_number, 0);
    init_array<float>(nrn_g_Na, neurons_number, 0);
    init_array<float>(nrn_g_K, neurons_number, 0);
    init_array<float>(nrn_g_L, neurons_number, 0);

    init_array<float>(CV1_spikes_ms, len, 0);
    init_array<float>(CV2_spikes_ms, len, 0);
    init_array<float>(CV3_spikes_ms, len, 0);
    init_array<float>(CV4_spikes_ms, len * 2, 0);
    init_array<float>(CV5_spikes_ms, len, 0);

    rand_normal_init_array<int>(nrn_ref_time, neurons_number, (int)(3 / SIM_STEP), (int)(0.4 / SIM_STEP));  // neuron ref time, aprx interval is (1.8, 4.2)
	rand_normal_init_array<float>(nrn_threshold, neurons_number, -50, 0.4); // neurons threshold (-51.2, -48.8)

    random_device r_s;
    default_random_engine generator_s(r_s());
    lognormal_distribution<float> distribution(0.6, 0.2); // 0.5 0.2
    // number of spikes per 1ms
    for (int i = 0; i < len; i++) {
        CV1_spikes_ms[i] = static_cast<int>((distribution(generator_s) - 1) * 25);
        CV2_spikes_ms[i] = static_cast<int>((distribution(generator_s) - 1) * 25);
        CV3_spikes_ms[i] = static_cast<int>((distribution(generator_s) - 1) * 25);
        CV5_spikes_ms[i] = static_cast<int>((distribution(generator_s) - 1) * 25);

    }

    for (int i = 0; i < len * 2; i++) {
        CV4_spikes_ms[i] = static_cast<int>((distribution(generator_s) - 1) * 25);
    }


    int CV1_spikes[SIM_TIME_IN_STEPS];
    int CV2_spikes[SIM_TIME_IN_STEPS];
    int CV3_spikes[SIM_TIME_IN_STEPS];
    int CV4_spikes[SIM_TIME_IN_STEPS];
    int CV5_spikes[SIM_TIME_IN_STEPS];


    init_array<int>(CV1_spikes, SIM_TIME_IN_STEPS, 0);
    init_array<int>(CV2_spikes, SIM_TIME_IN_STEPS, 0);
    init_array<int>(CV3_spikes, SIM_TIME_IN_STEPS, 0);
    init_array<int>(CV4_spikes, SIM_TIME_IN_STEPS, 0);
    init_array<int>(CV5_spikes, SIM_TIME_IN_STEPS, 0);


    int start = begin_C_spiking[0];
    int end = begin_C_spiking[0] + 40;
    for (int i = 0; i < len; i++) {
        int number_of_spikes = CV1_spikes_ms[i];
        uniform_int_distribution<int> distr(start, end);
        for (int j = 0; j < number_of_spikes; j++) {
            int pos = distr(generator_s);
            CV1_spikes[pos] = 1;
        }
        start += 40; end += 40;
    }

    start = begin_C_spiking[1];
    end = begin_C_spiking[1] + 40;
    for (int i = 0; i < len; i++) {
        int number_of_spikes = CV2_spikes_ms[i];
        uniform_int_distribution<int> distr(start, end);
        for (int j = 0; j < number_of_spikes; j++) {
            int pos = distr(generator_s);
            CV2_spikes[pos] = 1;
        }
        start += 40; end += 40;
    }

    start = begin_C_spiking[2];
    end = begin_C_spiking[2] + 40;
    for (int i = 0; i < len; i++) {
        int number_of_spikes = CV3_spikes_ms[i];
        uniform_int_distribution<int> distr(start, end);
        for (int j = 0; j < number_of_spikes; j++) {
            int pos = distr(generator_s);
            CV3_spikes[pos] = 1;
        }
        start += 40; end += 40;
    }

    start = begin_C_spiking[3];
    end = start + 80;
    for (int i = 0; i < len * 2; i++) {
        int number_of_spikes = CV4_spikes_ms[i];
        uniform_int_distribution<int> distr(start, end);
        for (int j = 0; j < number_of_spikes; j++) {
            int pos = distr(generator_s);
            CV4_spikes[pos] = 1;
        }
        start += 40; end += 40;
    }

    start = begin_C_spiking[4];
    end = start + 40;
    for (int i = 0; i < len; i++) {
        int number_of_spikes = CV5_spikes_ms[i];
        uniform_int_distribution<int> distr(start, end);
        for (int j = 0; j < number_of_spikes; j++) {
            int pos = distr(generator_s);
            CV5_spikes[pos] = 1;
        }
        start += 40; end += 40;
    }


	/// fill array of diameters for moto neurons
    bimodal_distr_for_moto_neurons(nrn_diameter);

    /// EES, E1, E2, E3, E4, E5
    for (int i = 0; i < 120; i++) {
        nrn_diameter[i] = 5;
    }

	/// fill array of diameters for inter neurons
    random_device r1;
    default_random_engine generator1(r1());
    uniform_real_distribution<float> d_inter_distr(1, 10);
    for (int i = 120; i < 1127; i++) {
        nrn_diameter[i] = d_inter_distr(generator1);
    }
    for (int i = 1705; i < neurons_number; i++) {
        nrn_diameter[i] = d_inter_distr(generator1);
    }

    /// fill array of diameters for Ia_aff neurons
    random_device r2;
    default_random_engine generator2(r2());
    uniform_real_distribution<float> d_Ia_aff_distr(10, 20);
    for (int i = 1465; i < 1705; i++) {
        nrn_diameter[i] = d_Ia_aff_distr(generator2);
    }

    /// fill C_m, g_Na, g_K, g_L arrays
    for(int i = 0; i < neurons_number; i++) {
        float d = nrn_diameter[i];
        float S = M_PI * d * d;
        float s_cM = 0.01;
        if(i >= 1127 && i <= 1464) {
            s_cM = 0.02;
        }
        nrn_c_m[i] = S * s_cM;
        nrn_g_Na[i] = S * 0.5;
        nrn_g_K[i] = S * 3;
        nrn_g_L[i] = S * 0.02;
    }


	// synapse variables
	auto *synapses_pre_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	auto *synapses_post_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	auto *synapses_weight = (float *) malloc(datasize<float>(synapses_number));
	auto *synapses_delay = (int *) malloc(datasize<int>(synapses_number));
	auto *synapses_delay_timer = (int *) malloc(datasize<int>(synapses_number));
	init_array<int>(synapses_delay_timer, synapses_number, -1);

	// fill arrays of synapses
	unsigned int syn_id = 0;
	for(SynapseMetadata metadata : all_synapses) {
		synapses_pre_nrn_id[syn_id] = metadata.pre_id;
		synapses_post_nrn_id[syn_id] = metadata.post_id;
		synapses_delay[syn_id] = metadata.synapse_delay;
		synapses_weight[syn_id] = metadata.synapse_weight;
		syn_id++;
	}
	all_synapses.clear();

	// neuron variables
	float* gpu_nrn_n;
	float* gpu_nrn_h;
	float* gpu_nrn_m;
	float* gpu_nrn_c_m;
	float* gpu_nrn_v_m;
	float* gpu_nrn_g_exc;
	float* gpu_nrn_g_inh;
	float* gpu_nrn_threshold;
	bool* gpu_nrn_has_spike;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_time_timer;
	float* gpu_nrn_g_Na;
	float* gpu_nrn_g_K;
	float* gpu_nrn_g_L;

	// synapse variables
	int* gpu_syn_pre_nrn_id;
	int* gpu_syn_post_nrn_id;
	float* gpu_syn_weight;
	int* gpu_syn_delay;
	int* gpu_syn_delay_timer;

	int *gpu_begin_C_spiking;
	int *gpu_end_C_spiking;

	int *gpu_spikes_CV1;
    int *gpu_spikes_CV2;
    int *gpu_spikes_CV3;
    int *gpu_spikes_CV4;
    int *gpu_spikes_CV5;

    // allocate memory in the GPU
	cudaMalloc(&gpu_nrn_n, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_h, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_v_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_c_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_g_exc, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_g_inh, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_threshold, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_has_spike, datasize<bool>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time_timer, datasize<int>(neurons_number));

    cudaMalloc(&gpu_nrn_g_Na, datasize<float>(neurons_number));
    cudaMalloc(&gpu_nrn_g_K, datasize<float>(neurons_number));
    cudaMalloc(&gpu_nrn_g_L, datasize<float>(neurons_number));

    cudaMalloc(&gpu_syn_pre_nrn_id, datasize<int>(synapses_number));
	cudaMalloc(&gpu_syn_post_nrn_id, datasize<int>(synapses_number));
	cudaMalloc(&gpu_syn_weight, datasize<float>(synapses_number));
	cudaMalloc(&gpu_syn_delay, datasize<int>(synapses_number));
	cudaMalloc(&gpu_syn_delay_timer, datasize<int>(synapses_number));

	cudaMalloc(&gpu_begin_C_spiking, datasize<int>(5));
	cudaMalloc(&gpu_end_C_spiking, datasize<int>(5));

    cudaMalloc(&gpu_spikes_CV1, datasize<int>(SIM_TIME_IN_STEPS));
    cudaMalloc(&gpu_spikes_CV2, datasize<int>(SIM_TIME_IN_STEPS));
    cudaMalloc(&gpu_spikes_CV3, datasize<int>(SIM_TIME_IN_STEPS));
    cudaMalloc(&gpu_spikes_CV4, datasize<int>(SIM_TIME_IN_STEPS));
    cudaMalloc(&gpu_spikes_CV5, datasize<int>(SIM_TIME_IN_STEPS));


    // copy data from CPU to GPU
	memcpyHtD<float>(gpu_nrn_n, nrn_n, neurons_number);
	memcpyHtD<float>(gpu_nrn_h, nrn_h, neurons_number);
	memcpyHtD<float>(gpu_nrn_m, nrn_m, neurons_number);
	memcpyHtD<float>(gpu_nrn_v_m, nrn_v_m, neurons_number);
	memcpyHtD<float>(gpu_nrn_c_m, nrn_c_m, neurons_number);
	memcpyHtD<float>(gpu_nrn_g_exc, nrn_g_exc, neurons_number);
	memcpyHtD<float>(gpu_nrn_g_inh, nrn_g_inh, neurons_number);
	memcpyHtD<float>(gpu_nrn_threshold, nrn_g_inh, neurons_number);
	memcpyHtD<bool>(gpu_nrn_has_spike, nrn_has_spike, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time_timer, nrn_ref_time_timer, neurons_number);

    memcpyHtD<float>(gpu_nrn_g_Na, nrn_g_Na, neurons_number);
    memcpyHtD<float>(gpu_nrn_g_K, nrn_g_K, neurons_number);
    memcpyHtD<float>(gpu_nrn_g_L, nrn_g_L, neurons_number);

    memcpyHtD<int>(gpu_syn_pre_nrn_id, synapses_pre_nrn_id, synapses_number);
	memcpyHtD<int>(gpu_syn_post_nrn_id, synapses_post_nrn_id, synapses_number);
	memcpyHtD<float>(gpu_syn_weight, synapses_weight, synapses_number);
	memcpyHtD<int>(gpu_syn_delay, synapses_delay, synapses_number);
	memcpyHtD<int>(gpu_syn_delay_timer, synapses_delay_timer, synapses_number);

	memcpyHtD<int>(gpu_begin_C_spiking, begin_C_spiking, 5);
	memcpyHtD<int>(gpu_end_C_spiking, end_C_spiking, 5);

    memcpyHtD<int>(gpu_spikes_CV1, CV1_spikes, SIM_TIME_IN_STEPS);
    memcpyHtD<int>(gpu_spikes_CV2, CV2_spikes, SIM_TIME_IN_STEPS);
    memcpyHtD<int>(gpu_spikes_CV3, CV3_spikes, SIM_TIME_IN_STEPS);
    memcpyHtD<int>(gpu_spikes_CV4, CV4_spikes, SIM_TIME_IN_STEPS);
    memcpyHtD<int>(gpu_spikes_CV5, CV5_spikes, SIM_TIME_IN_STEPS);


    // preparations for simulation
	int threads_per_block = 32;
	unsigned int nrn_num_blocks = neurons_number / threads_per_block + 1;
	unsigned int syn_num_blocks = synapses_number / threads_per_block + 1;

	auto total_nrn_threads = threads_per_block * nrn_num_blocks;
	auto total_syn_threads = threads_per_block * syn_num_blocks;

	printf("* * * Start GPU * * *\n");
	printf("Neurons kernel: %d threads (%.3f%% threads idle) [%d blocks x %d threads per block] mapped on %d neurons \n",
	       total_nrn_threads, (double)(total_nrn_threads - neurons_number) / total_nrn_threads * 100,
	       nrn_num_blocks, threads_per_block, neurons_number);
	printf("Synapses kernel: %d threads (%.3f%% threads idle) [%d blocks x %d threads per block] mapped on %d synapses \n",
	       total_syn_threads, (double)(total_syn_threads - synapses_number) / total_syn_threads * 100,
	       syn_num_blocks, threads_per_block, synapses_number);

	// stuff variables for controlling C0/C1 activation
	int local_iter = 0;
	int activated_C_ = 1; // start from extensor
	int early_activated_C_ = 1;
	int shift_time_by_step = 0;
	int decrease_lvl_Ia_spikes;

	simulation_t_start = chrono::system_clock::now();

	// the main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		decrease_lvl_Ia_spikes = 0;
		// if flexor C0 activated, find the end of it and change to C1
		if (activated_C_ == 0) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				activated_C_ = 1; // change to C1
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C0;  // add constant 125 ms
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C0 == 0) {
				early_activated_C_ = 1;
			}
			// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				activated_C_ = 0; // change to C0
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C1;  // add time equal to n_layers * 25 ms
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C1 == 0) {
				early_activated_C_ = 0;
			}
		}

		int shifted_iter_time = sim_iter - shift_time_by_step;

		// CV1
		if ((begin_C_spiking[0] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[0])) {
			decrease_lvl_Ia_spikes = 2;
		} else {
			// CV2
			if ((begin_C_spiking[1] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[1])) {
				decrease_lvl_Ia_spikes = 1;
			} else {
				// CV3
				if ((begin_C_spiking[2] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[2])) {
					decrease_lvl_Ia_spikes = 0;
				} else {
					// CV4
					if ((begin_C_spiking[3] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[3])) {
						decrease_lvl_Ia_spikes = 1;
					} else {
						// CV5
						if ((begin_C_spiking[4] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[4])) {
							decrease_lvl_Ia_spikes = 2;
						}
					}
				}
			}
		}


		// update local iter (warning: can be resetted at C0/C1 activation)
		local_iter++;

		// invoke GPU kernel for neurons
        neurons_kernel<<<nrn_num_blocks, threads_per_block>>>(gpu_nrn_c_m,
                gpu_nrn_v_m,
                gpu_nrn_h,
                gpu_nrn_m,
                gpu_nrn_n,
                gpu_nrn_g_exc,
                gpu_nrn_g_inh,
                gpu_nrn_threshold,
                gpu_nrn_has_spike,
                gpu_nrn_ref_time,
                gpu_nrn_ref_time_timer,
                neurons_number,
                sim_iter - shift_time_by_step,
                activated_C_,
                early_activated_C_,
                sim_iter,
                gpu_begin_C_spiking,
                gpu_end_C_spiking,
                decrease_lvl_Ia_spikes,
                ees_spike_each_step,
                gpu_nrn_g_Na,
                gpu_nrn_g_K,
                gpu_nrn_g_L);

		// copy data from GPU
		memcpyDtH<float>(nrn_v_m, gpu_nrn_v_m, neurons_number);
		memcpyDtH<float>(nrn_g_exc, gpu_nrn_g_exc, neurons_number);
		memcpyDtH<float>(nrn_g_inh, gpu_nrn_g_inh, neurons_number);
		memcpyDtH<bool>(nrn_has_spike, gpu_nrn_has_spike, neurons_number);

		// fill records arrays
		for(GroupMetadata &metadata : all_groups) {
			if (save_all == 0) {
				if (metadata.group.group_name == "MN_E")
					copy_data_to(metadata, nrn_v_m, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
				if (metadata.group.group_name == "MN_F")
					copy_data_to(metadata, nrn_v_m, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
			} else
				copy_data_to(metadata, nrn_v_m, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
		}

		// invoke GPU kernel for synapses
		synapses_kernel<<<syn_num_blocks, threads_per_block>>>(gpu_nrn_has_spike,
		                                                       gpu_nrn_g_exc,
		                                                       gpu_nrn_g_inh,
		                                                       gpu_syn_pre_nrn_id,
		                                                       gpu_syn_post_nrn_id,
		                                                       gpu_syn_delay,
		                                                       gpu_syn_delay_timer,
		                                                       gpu_syn_weight,
		                                                       synapses_number);
	} // end of the simulation iteration loop

	simulation_t_end = chrono::system_clock::now();

	cudaDeviceSynchronize();  // tell the CPU to halt further processing until the CUDA has finished doing its business
	cudaDeviceReset();        // remove all all device allocations (destroy a CUDA context)

	// save recorded data
	save_result(itest, save_all);

	auto sim_time_diff = chrono::duration_cast<chrono::milliseconds>(simulation_t_end - simulation_t_start).count();
	printf("Elapsed %li ms (measured) | T_sim = %d ms\n", sim_time_diff, T_simulation);
	printf("%s x%f\n", (float)(T_simulation / sim_time_diff) > 1?
	       COLOR_GREEN "faster" COLOR_RESET: COLOR_RED "slower" COLOR_RESET, (float)T_simulation / sim_time_diff);
}


// runner
int main(int argc, char* argv[]) {

    int cms = 21;
    int ees = 40;
    float inh = 1;
    bool ped = false;
    bool ht5 = false;
    int save_all = 0;
    int itest = 0;

    float quadru_coef = ped ? 0.5 : 1;
    float sero_coef = ht5 ? 5.3 : 1;

    ///////////////////////////

    /// from EES
    float E1_OM1_0 = atof(argv[1]);
    float E2_OM2_0 = atof(argv[2]);
    float E3_OM3_0 = atof(argv[3]);
    float E4_OM4_0 = atof(argv[4]);
    float E5_OM5_0 = atof(argv[5]);

    /// CV - OM1
    float CV1_OM1_0 = atof(argv[6]) * quadru_coef * sero_coef;
    float CV2_OM1_0 = atof(argv[7]) * quadru_coef * sero_coef;

    /// CV - OM2
    float CV2_OM2_0 = atof(argv[8]) * quadru_coef * sero_coef;
    float CV3_OM2_0 = atof(argv[9]) * quadru_coef * sero_coef;

    /// CV - OM3
    float CV3_OM3_0 = atof(argv[10]) * quadru_coef * sero_coef;
    float CV4_OM3_0 = atof(argv[11]) * quadru_coef * sero_coef;

    /// CV - OM4
    float CV4_OM4_0 = atof(argv[12]) * quadru_coef * sero_coef;
    float CV5_OM4_0 = atof(argv[13]) * quadru_coef * sero_coef;

    /// CV - OM5
    float CV5_OM5_0 = atof(argv[14]) * quadru_coef * sero_coef;

    ///
    float CV3_OM1_3 = atof(argv[15]);
    float CV4_OM1_3 = atof(argv[16]);
    float CV5_OM1_3 = atof(argv[17]);
    float CV4_OM2_3 = atof(argv[18]);
    float CV5_OM2_3 = atof(argv[19]);
    float CV5_OM3_3 = atof(argv[20]);

    /// OM1
    float OM1_0_OM1_1 = atof(argv[21]);
    float OM1_1_OM1_2_E = atof(argv[22]);
    float OM1_1_OM1_2_F = atof(argv[23]);
    float OM1_1_OM1_3 = atof(argv[24]);
    float OM1_2_E_OM1_1 = atof(argv[25]);
    float OM1_2_F_OM1_1 = atof(argv[26]);
    float OM1_2_E_OM1_3 = atof(argv[27]);
    float OM1_2_F_OM1_3 = atof(argv[28]);
    float OM1_3_OM1_1 = atof(argv[29]);
    float OM1_3_OM1_2_E = atof(argv[30]);
    float OM1_3_OM1_2_F = atof(argv[31]);

    float OM1_2_E_eIP_E = atof(argv[32]);
    float OM1_2_F_eIP_F = atof(argv[33]);
    float OM1_2_F_OM2_2_F = atof(argv[34]);

    /// OM2
    float OM2_0_OM2_1 = atof(argv[35]);
    float OM2_1_OM2_2_E = atof(argv[36]);
    float OM2_1_OM2_2_F = atof(argv[37]);
    float OM2_1_OM2_3 = atof(argv[38]);
    float OM2_2_E_OM2_1 = atof(argv[39]);
    float OM2_2_F_OM2_1 = atof(argv[40]);
    float OM2_2_E_OM2_3 = atof(argv[41]);
    float OM2_2_F_OM2_3 = atof(argv[42]);
    float OM2_3_OM2_1 = atof(argv[43]);
    float OM2_3_OM2_2_E = atof(argv[44]);
    float OM2_3_OM2_2_F = atof(argv[45]);

    float OM2_2_E_eIP_E = atof(argv[46]);
    float OM2_2_F_eIP_F = atof(argv[47]);
    float OM2_2_F_OM3_2_F = atof(argv[48]);

    /// OM3
    float OM3_0_OM3_1 = atof(argv[49]);
    float OM3_1_OM3_2_E = atof(argv[50]);
    float OM3_1_OM3_2_F = atof(argv[51]);
    float OM3_1_OM3_3 = atof(argv[52]);
    float OM3_2_E_OM3_1 = atof(argv[53]);
    float OM3_2_F_OM3_1 = atof(argv[54]);
    float OM3_2_E_OM3_3 = atof(argv[55]);
    float OM3_2_F_OM3_3 = atof(argv[56]);
    float OM3_3_OM3_1 = atof(argv[57]);
    float OM3_3_OM3_2_E = atof(argv[58]);
    float OM3_3_OM3_2_F = atof(argv[59]);

    float OM3_2_E_eIP_E = atof(argv[60]);
    float OM3_2_F_eIP_F = atof(argv[61]);
    float OM3_2_F_OM4_2_F = atof(argv[62]);

    /// OM4
    float OM4_0_OM4_1 = atof(argv[63]);
    float OM4_1_OM4_2_E = atof(argv[64]);
    float OM4_1_OM4_2_F = atof(argv[65]);
    float OM4_1_OM4_3 = atof(argv[66]);
    float OM4_2_E_OM4_1 = atof(argv[67]);
    float OM4_2_F_OM4_1 = atof(argv[68]);
    float OM4_2_E_OM4_3 = atof(argv[69]);
    float OM4_2_F_OM4_3 = atof(argv[70]);
    float OM4_3_OM4_1 = atof(argv[71]);
    float OM4_3_OM4_2_E = atof(argv[72]);
    float OM4_3_OM4_2_F = atof(argv[73]);

    float OM4_2_E_eIP_E = atof(argv[74]);
    float OM4_2_F_eIP_F = atof(argv[75]);
    float OM4_2_F_OM5_2_F = atof(argv[76]);

    /// OM5
    float OM5_0_OM5_1 = atof(argv[77]);
    float OM5_1_OM5_2_E = atof(argv[78]);
    float OM5_1_OM5_2_F = atof(argv[79]);
    float OM5_1_OM5_3 = atof(argv[80]);
    float OM5_2_E_OM5_1 = atof(argv[81]);
    float OM5_2_F_OM5_1 = atof(argv[82]);
    float OM5_2_E_OM5_3 = atof(argv[83]);
    float OM5_2_F_OM5_3 = atof(argv[84]);
    float OM5_3_OM5_1 = atof(argv[85]);
    float OM5_3_OM5_2_E = atof(argv[86]);
    float OM5_3_OM5_2_F = atof(argv[87]);

    float OM5_2_E_eIP_E = atof(argv[88]);
    float OM5_2_F_eIP_F = atof(argv[89]);

    /// reflex arc
    float CV1_iIP_E = atof(argv[90]);
    float CV2_iIP_E = atof(argv[91]);
    float CV3_iIP_E = atof(argv[92]);
    float CV4_iIP_E = atof(argv[93]);
    float CV5_iIP_E = atof(argv[94]);

    float iIP_E_eIP_F = atof(argv[95]);
    float iIP_F_eIP_E = atof(argv[96]);
    float iIP_E_OM1_2_F = atof(argv[97]);
    float iIP_E_OM2_2_F = atof(argv[98]);
    float iIP_E_OM3_2_F = atof(argv[99]);
    float iIP_E_OM4_2_F = atof(argv[100]);
    float eIP_E_MN_E = atof(argv[101]);
    float eIP_F_MN_F = atof(argv[102]);
    float iIP_E_Ia_E_pool = atof(argv[103]);
    float iIP_F_Ia_F_pool = atof(argv[104]);
    float Ia_E_pool_MN_F = atof(argv[105]);
    float Ia_E_pool_Ia_F_pool = atof(argv[106]);
    float Ia_F_pool_MN_E = atof(argv[107]);
    float Ia_F_pool_Ia_E_pool = atof(argv[108]);
    float Ia_E_aff_MN_E = atof(argv[109]);
    float Ia_F_aff_MN_F = atof(argv[110]);
    float MN_E_R_E = atof(argv[111]);
    float MN_F_R_F = atof(argv[112]);
    float R_E_MN_E = atof(argv[113]);
    float R_E_R_F = atof(argv[114]);
    float R_F_MN_F = atof(argv[115]);
    float R_F_R_E = atof(argv[116]);

    //// delays

    /// from EES
    float E1_OM1_0_d = atof(argv[117]);
    float E2_OM2_0_d = atof(argv[118]);
    float E3_OM3_0_d = atof(argv[119]);
    float E4_OM4_0_d = atof(argv[120]);
    float E5_OM5_0_d = atof(argv[121]);

    /// CV - OM1
    float CV1_OM1_0_d = atof(argv[122]);
    float CV2_OM1_0_d = atof(argv[123]);
    float CV3_OM1_3_d = atof(argv[124]);
    float CV4_OM1_3_d = atof(argv[125]);
    float CV5_OM1_3_d = atof(argv[126]);

    /// CV - OM2
    float CV2_OM2_0_d = atof(argv[127]);
    float CV3_OM2_0_d = atof(argv[128]);
    float CV4_OM2_3_d = atof(argv[129]);
    float CV5_OM2_3_d = atof(argv[130]);

    /// CV - OM3
    float CV3_OM3_0_d = atof(argv[131]);
    float CV4_OM3_0_d = atof(argv[132]);
    float CV5_OM3_3_d = atof(argv[133]);

    /// CV - OM4
    float CV4_OM4_0_d = atof(argv[134]);
    float CV5_OM4_0_d = atof(argv[135]);

    /// CV - OM5
    float CV5_OM5_0_d = atof(argv[136]);

    /// OM1
    float OM1_0_OM1_1_d = atof(argv[137]);
    float OM1_1_OM1_2_E_d = atof(argv[138]);
    float OM1_1_OM1_2_F_d = atof(argv[139]);
    float OM1_1_OM1_3_d = atof(argv[140]);
    float OM1_2_E_OM1_1_d = atof(argv[141]);
    float OM1_2_F_OM1_1_d = atof(argv[142]);
    float OM1_2_E_OM1_3_d = atof(argv[143]);
    float OM1_2_F_OM1_3_d = atof(argv[144]);
    float OM1_3_OM1_1_d = atof(argv[145]);
    float OM1_3_OM1_2_E_d = atof(argv[146]);
    float OM1_3_OM1_2_F_d = atof(argv[147]);

    float OM1_2_E_eIP_E_d = atof(argv[148]);
    float OM1_2_F_eIP_F_d = atof(argv[149]);
    float OM1_2_F_OM2_2_F_d = atof(argv[150]);

    /// OM2
    float OM2_0_OM2_1_d = atof(argv[151]);
    float OM2_1_OM2_2_E_d = atof(argv[152]);
    float OM2_1_OM2_2_F_d = atof(argv[153]);
    float OM2_1_OM2_3_d = atof(argv[154]);
    float OM2_2_E_OM2_1_d = atof(argv[155]);
    float OM2_2_F_OM2_1_d = atof(argv[156]);
    float OM2_2_E_OM2_3_d = atof(argv[157]);
    float OM2_2_F_OM2_3_d = atof(argv[158]);
    float OM2_3_OM2_1_d = atof(argv[159]);
    float OM2_3_OM2_2_E_d = atof(argv[160]);
    float OM2_3_OM2_2_F_d = atof(argv[161]);

    float OM2_2_E_eIP_E_d = atof(argv[162]);
    float OM2_2_F_eIP_F_d = atof(argv[163]);
    float OM2_2_F_OM3_2_F_d = atof(argv[164]);

    /// OM3
    float OM3_0_OM3_1_d = atof(argv[165]);
    float OM3_1_OM3_2_E_d = atof(argv[166]);
    float OM3_1_OM3_2_F_d = atof(argv[167]);
    float OM3_1_OM3_3_d = atof(argv[168]);
    float OM3_2_E_OM3_1_d = atof(argv[169]);
    float OM3_2_F_OM3_1_d = atof(argv[170]);
    float OM3_2_E_OM3_3_d = atof(argv[171]);
    float OM3_2_F_OM3_3_d = atof(argv[172]);
    float OM3_3_OM3_1_d = atof(argv[173]);
    float OM3_3_OM3_2_E_d = atof(argv[174]);
    float OM3_3_OM3_2_F_d = atof(argv[175]);

    float OM3_2_E_eIP_E_d = atof(argv[176]);
    float OM3_2_F_eIP_F_d = atof(argv[177]);
    float OM3_2_F_OM4_2_F_d = atof(argv[178]);

    /// OM4
    float OM4_0_OM4_1_d = atof(argv[179]);
    float OM4_1_OM4_2_E_d = atof(argv[180]);
    float OM4_1_OM4_2_F_d = atof(argv[181]);
    float OM4_1_OM4_3_d = atof(argv[182]);
    float OM4_2_E_OM4_1_d = atof(argv[183]);
    float OM4_2_F_OM4_1_d = atof(argv[184]);
    float OM4_2_E_OM4_3_d = atof(argv[185]);
    float OM4_2_F_OM4_3_d = atof(argv[186]);
    float OM4_3_OM4_1_d = atof(argv[187]);
    float OM4_3_OM4_2_E_d = atof(argv[188]);
    float OM4_3_OM4_2_F_d = atof(argv[189]);

    float OM4_2_E_eIP_E_d = atof(argv[190]);
    float OM4_2_F_eIP_F_d = atof(argv[191]);
    float OM4_2_F_OM5_2_F_d = atof(argv[192]);

    /// OM5
    float OM5_0_OM5_1_d = atof(argv[193]);
    float OM5_1_OM5_2_E_d = atof(argv[194]);
    float OM5_1_OM5_2_F_d = atof(argv[195]);
    float OM5_1_OM5_3_d = atof(argv[196]);
    float OM5_2_E_OM5_1_d = atof(argv[197]);
    float OM5_2_F_OM5_1_d = atof(argv[198]);
    float OM5_2_E_OM5_3_d = atof(argv[199]);
    float OM5_2_F_OM5_3_d = atof(argv[200]);
    float OM5_3_OM5_1_d = atof(argv[201]);
    float OM5_3_OM5_2_E_d = atof(argv[202]);
    float OM5_3_OM5_2_F_d = atof(argv[203]);

    float OM5_2_E_eIP_E_d = atof(argv[204]);
    float OM5_2_F_eIP_F_d = atof(argv[205]);

    float CV1_iIP_E_d = atof(argv[206]);
    float CV2_iIP_E_d = atof(argv[207]);
    float CV3_iIP_E_d = atof(argv[208]);
    float CV4_iIP_E_d = atof(argv[209]);
    float CV5_iIP_E_d = atof(argv[210]);

    float iIP_E_eIP_F_d = atof(argv[211]);
    float iIP_F_eIP_E_d = atof(argv[212]);
    float iIP_E_OM1_2_F_d = atof(argv[213]);
    float iIP_E_OM2_2_F_d = atof(argv[214]);
    float iIP_E_OM3_2_F_d = atof(argv[215]);
    float iIP_E_OM4_2_F_d = atof(argv[216]);
    float eIP_E_MN_E_d = atof(argv[217]);
    float eIP_F_MN_F_d = atof(argv[218]);
    float iIP_E_Ia_E_pool_d = atof(argv[219]);
    float iIP_F_Ia_F_pool_d = atof(argv[220]);
    float Ia_E_pool_MN_F_d = atof(argv[221]);
    float Ia_E_pool_Ia_F_pool_d = atof(argv[222]);
    float Ia_F_pool_MN_E_d = atof(argv[223]);
    float Ia_F_pool_Ia_E_pool_d = atof(argv[224]);
    float Ia_E_aff_MN_E_d = atof(argv[225]);
    float Ia_F_aff_MN_F_d = atof(argv[226]);
    float MN_E_R_E_d = atof(argv[227]);
    float MN_F_R_F_d = atof(argv[228]);
    float R_E_MN_E_d = atof(argv[229]);
    float R_E_R_F_d = atof(argv[230]);
    float R_F_MN_F_d = atof(argv[231]);
    float R_F_R_E_d = atof(argv[232]);


	simulate(cms, ees, inh, ped, ht5, save_all, itest,
             E1_OM1_0,
             E2_OM2_0,
             E3_OM3_0,
             E4_OM4_0,
             E5_OM5_0,
             CV1_OM1_0,
             CV2_OM1_0,
             CV3_OM1_3,
             CV4_OM1_3,
             CV5_OM1_3,
             CV2_OM2_0,
             CV3_OM2_0,
             CV4_OM2_3,
             CV5_OM2_3,
             CV3_OM3_0,
             CV4_OM3_0,
             CV5_OM3_3,
             CV4_OM4_0,
             CV5_OM4_0,
             CV5_OM5_0,
             OM1_0_OM1_1,
             OM1_1_OM1_2_E,
             OM1_1_OM1_2_F,
             OM1_1_OM1_3,
             OM1_2_E_OM1_1,
             OM1_2_F_OM1_1,
             OM1_2_E_OM1_3,
             OM1_2_F_OM1_3,
             OM1_3_OM1_1,
             OM1_3_OM1_2_E,
             OM1_3_OM1_2_F,
             OM1_2_E_eIP_E,
             OM1_2_F_eIP_F,
             OM1_2_F_OM2_2_F,
             OM2_0_OM2_1,
             OM2_1_OM2_2_E,
             OM2_1_OM2_2_F,
             OM2_1_OM2_3,
             OM2_2_E_OM2_1,
             OM2_2_F_OM2_1,
             OM2_2_E_OM2_3,
             OM2_2_F_OM2_3,
             OM2_3_OM2_1,
             OM2_3_OM2_2_E,
             OM2_3_OM2_2_F,
             OM2_2_E_eIP_E,
             OM2_2_F_eIP_F,
             OM2_2_F_OM3_2_F,
             OM3_0_OM3_1,
             OM3_1_OM3_2_E,
             OM3_1_OM3_2_F,
             OM3_1_OM3_3,
             OM3_2_E_OM3_1,
             OM3_2_F_OM3_1,
             OM3_2_E_OM3_3,
             OM3_2_F_OM3_3,
             OM3_3_OM3_1,
             OM3_3_OM3_2_E,
             OM3_3_OM3_2_F,
             OM3_2_E_eIP_E,
             OM3_2_F_eIP_F,
             OM3_2_F_OM4_2_F,
             OM4_0_OM4_1,
             OM4_1_OM4_2_E,
             OM4_1_OM4_2_F,
             OM4_1_OM4_3,
             OM4_2_E_OM4_1,
             OM4_2_F_OM4_1,
             OM4_2_E_OM4_3,
             OM4_2_F_OM4_3,
             OM4_3_OM4_1,
             OM4_3_OM4_2_E,
             OM4_3_OM4_2_F,
             OM4_2_E_eIP_E,
             OM4_2_F_eIP_F,
             OM4_2_F_OM5_2_F,
             OM5_0_OM5_1,
             OM5_1_OM5_2_E,
             OM5_1_OM5_2_F,
             OM5_1_OM5_3,
             OM5_2_E_OM5_1,
             OM5_2_F_OM5_1,
             OM5_2_E_OM5_3,
             OM5_2_F_OM5_3,
             OM5_3_OM5_1,
             OM5_3_OM5_2_E,
             OM5_3_OM5_2_F,
             OM5_2_E_eIP_E,
             OM5_2_F_eIP_F,
             CV1_iIP_E,
             CV2_iIP_E,
             CV3_iIP_E,
             CV4_iIP_E,
             CV5_iIP_E,
             iIP_E_eIP_F,
             iIP_F_eIP_E,
             iIP_E_OM1_2_F,
             iIP_E_OM2_2_F,
             iIP_E_OM3_2_F,
             iIP_E_OM4_2_F,
             eIP_E_MN_E,
             eIP_F_MN_F,
             iIP_E_Ia_E_pool,
             iIP_F_Ia_F_pool,
             Ia_E_pool_MN_F,
             Ia_E_pool_Ia_F_pool,
             Ia_F_pool_MN_E,
             Ia_F_pool_Ia_E_pool,
             Ia_E_aff_MN_E,
             Ia_F_aff_MN_F,
             MN_E_R_E,
             MN_F_R_F,
             R_E_MN_E,
             R_E_R_F,
             R_F_MN_F,
             R_F_R_E,
             E1_OM1_0_d,
             E2_OM2_0_d,
             E3_OM3_0_d,
             E4_OM4_0_d,
             E5_OM5_0_d,
             CV1_OM1_0_d,
             CV2_OM1_0_d,
             CV3_OM1_3_d,
             CV4_OM1_3_d,
             CV5_OM1_3_d,
             CV2_OM2_0_d,
             CV3_OM2_0_d,
             CV4_OM2_3_d,
             CV5_OM2_3_d,
             CV3_OM3_0_d,
             CV4_OM3_0_d,
             CV5_OM3_3_d,
             CV4_OM4_0_d,
             CV5_OM4_0_d,
             CV5_OM5_0_d,
             OM1_0_OM1_1_d,
             OM1_1_OM1_2_E_d,
             OM1_1_OM1_2_F_d,
             OM1_1_OM1_3_d,
             OM1_2_E_OM1_1_d,
             OM1_2_F_OM1_1_d,
             OM1_2_E_OM1_3_d,
             OM1_2_F_OM1_3_d,
             OM1_3_OM1_1_d,
             OM1_3_OM1_2_E_d,
             OM1_3_OM1_2_F_d,
             OM1_2_E_eIP_E_d,
             OM1_2_F_eIP_F_d,
             OM1_2_F_OM2_2_F_d,
             OM2_0_OM2_1_d,
             OM2_1_OM2_2_E_d,
             OM2_1_OM2_2_F_d,
             OM2_1_OM2_3_d,
             OM2_2_E_OM2_1_d,
             OM2_2_F_OM2_1_d,
             OM2_2_E_OM2_3_d,
             OM2_2_F_OM2_3_d,
             OM2_3_OM2_1_d,
             OM2_3_OM2_2_E_d,
             OM2_3_OM2_2_F_d,
             OM2_2_E_eIP_E_d,
             OM2_2_F_eIP_F_d,
             OM2_2_F_OM3_2_F_d,
             OM3_0_OM3_1_d,
             OM3_1_OM3_2_E_d,
             OM3_1_OM3_2_F_d,
             OM3_1_OM3_3_d,
             OM3_2_E_OM3_1_d,
             OM3_2_F_OM3_1_d,
             OM3_2_E_OM3_3_d,
             OM3_2_F_OM3_3_d,
             OM3_3_OM3_1_d,
             OM3_3_OM3_2_E_d,
             OM3_3_OM3_2_F_d,
             OM3_2_E_eIP_E_d,
             OM3_2_F_eIP_F_d,
             OM3_2_F_OM4_2_F_d,
             OM4_0_OM4_1_d,
             OM4_1_OM4_2_E_d,
             OM4_1_OM4_2_F_d,
             OM4_1_OM4_3_d,
             OM4_2_E_OM4_1_d,
             OM4_2_F_OM4_1_d,
             OM4_2_E_OM4_3_d,
             OM4_2_F_OM4_3_d,
             OM4_3_OM4_1_d,
             OM4_3_OM4_2_E_d,
             OM4_3_OM4_2_F_d,
             OM4_2_E_eIP_E_d,
             OM4_2_F_eIP_F_d,
             OM4_2_F_OM5_2_F_d,
             OM5_0_OM5_1_d,
             OM5_1_OM5_2_E_d,
             OM5_1_OM5_2_F_d,
             OM5_1_OM5_3_d,
             OM5_2_E_OM5_1_d,
             OM5_2_F_OM5_1_d,
             OM5_2_E_OM5_3_d,
             OM5_2_F_OM5_3_d,
             OM5_3_OM5_1_d,
             OM5_3_OM5_2_E_d,
             OM5_3_OM5_2_F_d,
             OM5_2_E_eIP_E_d,
             OM5_2_F_eIP_F_d,
             CV1_iIP_E_d,
             CV2_iIP_E_d,
             CV3_iIP_E_d,
             CV4_iIP_E_d,
             CV5_iIP_E_d,
             iIP_E_eIP_F_d,
             iIP_F_eIP_E_d,
             iIP_E_OM1_2_F_d,
             iIP_E_OM2_2_F_d,
             iIP_E_OM3_2_F_d,
             iIP_E_OM4_2_F_d,
             eIP_E_MN_E_d,
             eIP_F_MN_F_d,
             iIP_E_Ia_E_pool_d,
             iIP_F_Ia_F_pool_d,
             Ia_E_pool_MN_F_d,
             Ia_E_pool_Ia_F_pool_d,
             Ia_F_pool_MN_E_d,
             Ia_F_pool_Ia_E_pool_d,
             Ia_E_aff_MN_E_d,
             Ia_F_aff_MN_F_d,
             MN_E_R_E_d,
             MN_F_R_F_d,
             R_E_MN_E_d,
             R_E_R_F_d,
             R_F_MN_F_d,
             R_F_R_E_d);

	return 0;
}