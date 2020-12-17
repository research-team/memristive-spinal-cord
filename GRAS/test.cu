#include <map>
#include <string>
#include <vector>
#include <iostream>

#define CHECK(call) {                                          \
    const cudaError_t error = call;                            \
    if (error != cudaSuccess) {                                \
        fprintf(stderr, "Error: %s:%d\ncode: %d\n%s\n",        \
        __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(error);                                           \
	}                                                          \
}
using namespace std;

// neuron parameters
struct Parameters {
	short *nrn_start_seg;   //
	char *models;           // [str] model's names
	float *Cm;              // [uF / cm2] membrane capacitance
	float *gnabar;          // [S / cm2] the maximal fast Na+ conductance
	float *gkbar;           // [S / cm2] the maximal slow K+ conductance
	float *gl;              // [S / cm2] the maximal leak conductance
	float *Ra;              // [Ohm cm] axoplasmic resistivity
	float *diam;            // [um] soma compartment diameter
	float *length;          // [um] soma compartment length
	float *ena;             // [mV] Na+ reversal (equilibrium, Nernst) potential
	float *ek;              // [mV] K+ reversal (equilibrium, Nernst) potential
	float *el;              // [mV] Leakage reversal (equilibrium) potential
	float *gkrect;          // [S / cm2] the maximal delayed rectifier K+ conductance
	float *gcaN;            // [S / cm2] the maximal N-type Ca2+ conductance
	float *gcaL;            // [S / cm2] the maximal L-type Ca2+ conductance
	float *gcak;            // [S / cm2] the maximal Ca2+ activated K+ conductance
	float *E_ex;            // [mV] excitatory reversal (equilibrium) potential
	float *E_inh;           // [mV] inhibitory reversal (equilibrium) potential
	float *tau_exc;         // [ms] rise time constant of excitatory synaptic conductance
	float *tau_inh1;        // [ms] rise time constant of inhibitory synaptic conductance
	float *tau_inh2;        // [ms] decay time constant of inhibitory synaptic conductance
	int size;
};

// neuron states
struct States {
	float *Vm;              // [mV] array for three compartments volatge
	float *n;               // [0..1] compartments channel, providing the kinetic pattern of the L conductance
	float *m;               // [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	float *h;               // [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	float *l;               // [0..1] inward rectifier potassium (Kir) channel
	float *s;               // [0..1] nodal slow potassium channel
	float *p;               // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	float *hc;              // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	float *mc;              // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	float *cai;             //
	float *I_Ca;            // [nA] Ca ionic currents
	float *NODE_A;          // the effect of this node on the parent node's equation
	float *NODE_B;          // the effect of the parent node on this node's equation
	float *NODE_D;          // diagonal element in node equation
	float *const_NODE_D;    // const diagonal element in node equation (performance)
	float *NODE_RHS;        // right hand side in node equation
	float *NODE_RINV;       // conductance uS from node to parent
	float *NODE_AREA;       // area of a node in um^2
	bool *has_spike;        // spike flag for each neuron
	bool *spike_on;         // special flag to prevent fake spike detecting
	float *g_exc;           // [S] excitatory conductivity level
	float *g_inh_A;         // [S] inhibitory conductivity level
	float *g_inh_B;         // [S] inhibitory conductivity level
	float *factor;          // [const] todo
	int size;
};

// copy data from host to device
template<typename type>
void memcpyHtD(type *gpu, type *host, unsigned int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

// copy data from device to host
template<typename type>
void memcpyDtH(type *host, type *gpu, unsigned int size) {
	cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

template<typename type>
type *init_gpu_arr(type *cpu_var, int size) {
	type *gpu_var;
	cudaMalloc(&gpu_var, sizeof(type) * size);
	memcpyHtD<type>(gpu_var, cpu_var, size);
	return gpu_var;
}

template<typename type>
type *init_cpu_arr(int size, type val) {
	type *array = new type[size];
	for (int i = 0; i < size; i++)
		array[i] = val;
	return array;
}

__global__
void some_kernel(States *S, Parameters *P, int size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		S->Vm[id] += 500;
		S->m[id] -= 100;
		S->n[id] += 1.45;
	}
}

int main(int argc, char **argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	const int size = 10000;
	// init structs
	States *dev_S, *S = (States *)malloc(sizeof(States));
	Parameters *dev_P, *P = (Parameters *)malloc(sizeof(Parameters));

	// Allocate and fill host data
	// [mV] neuron extracellular membrane potential
	auto *Vm = init_cpu_arr<float>(size, 0);
	auto *gpu_Vm = init_gpu_arr<float>(Vm, size);
//	float *gpu_Vm, *Vm = new float[size]();

	// init parameters
	short dev_nrn_start_seg, *nrn_start_seg = new short[size]();
	char dev_models, *models = new char[size]();
	float dev_Cm, *Cm = new float[size]();
	float dev_gnabar, *gnabar = new float[size]();
	float dev_gkbar, *gkbar = new float[size]();
	float dev_gl, *gl = new float[size]();
	float dev_Ra, *Ra = new float[size]();
	float dev_diam, *diam = new float[size]();
	float dev_length, *length = new float[size]();
	float dev_ena, *ena = new float[size]();
	float dev_ek, *ek = new float[size]();
	float dev_el, *el = new float[size]();
	float dev_gkrect, *gkrect = new float[size]();
	float dev_gcaN, *gcaN = new float[size]();
	float dev_gcaL, *gcaL = new float[size]();
	float dev_gcak, *gcak = new float[size]();
	float dev_E_ex, *E_ex = new float[size]();
	float dev_E_inh, *E_inh = new float[size]();
	float dev_tau_exc, *tau_exc = new float[size]();
	float dev_tau_inh1, *tau_inh1 = new float[size]();
	float dev_tau_inh2, *tau_inh2 = new float[size]();
	// init states
//	float *gpu_Vm, *Vm = new float[size]();
	float *dev_n, *n = new float[size]();
	float *dev_m, *m = new float[size]();
	float *dev_h, *h = new float[size]();
	float *dev_l, *l = new float[size]();
	float *dev_s, *s = new float[size]();
	float *dev_p, *p = new float[size]();
	float *dev_hc, *hc = new float[size]();
	float *dev_mc, *mc = new float[size]();
	float *dev_cai, *cai = new float[size]();
	float *dev_I_Ca, *I_Ca = new float[size]();
	float *dev_NODE_A, *NODE_A = new float[size]();
	float *dev_NODE_B, *NODE_B = new float[size]();
	float *dev_NODE_D, *NODE_D = new float[size]();
	float *dev_const_NODE_D, *const_NODE_D = new float[size]();
	float *dev_NODE_RHS, *NODE_RHS = new float[size]();
	float *dev_NODE_RINV, *NODE_RINV = new float[size]();
	float *dev_NODE_AREA, *NODE_AREA = new float[size]();
	bool *dev_has_spike, *has_spike = new bool[size]();
	bool *dev_spike_on, *spike_on = new bool[size]();
	float *dev_g_exc, *g_exc = new float[size]();
	float *dev_g_inh_A, *g_inh_A = new float[size]();
	float *dev_g_inh_B, *g_inh_B = new float[size]();
	float *dev_factor, *factor = new float[size]();

	for (int i = 0; i < size; i++) {
		Vm[i] = 500;
		m[i] = 200;
		n[i] = 10;
	}

	// Allocate device struct
	cudaMalloc((States **) &dev_S, sizeof(States));
	cudaMalloc((Parameters **) &dev_P, sizeof(Parameters));
	// Allocate device array pointers
	cudaMalloc((void **) &dev_n, size * sizeof(*dev_n));
	cudaMalloc((void **) &dev_m, size * sizeof(*dev_m));

	// Copy pointer content from host to device.
	cudaMemcpy(dev_n, n, size * sizeof(*n), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_m, m, size * sizeof(*m), cudaMemcpyHostToDevice);

	// Point to device pointer in host struct
	S->Vm = gpu_Vm;
	S->n = dev_n;
	S->m = dev_m;
	// Copy struct from host to device.
	cudaMemcpy(dev_S, S, sizeof(States), cudaMemcpyHostToDevice);

	// Call kernel
	some_kernel<<<10000/256 + 1, 256>>>(dev_S, dev_P, size); // block size need to be a multiply of 256
	CHECK(cudaDeviceSynchronize());

	// Copy result to host:
	cudaMemcpy(Vm, gpu_Vm, size * sizeof(*Vm), cudaMemcpyDeviceToHost);
	cudaMemcpy(m, dev_m, size * sizeof(*m), cudaMemcpyDeviceToHost);
	cudaMemcpy(n, dev_n, size * sizeof(*n), cudaMemcpyDeviceToHost);

	// Print some result
	std::cout << Vm[size-10] << std::endl;
	std::cout << m[size-10] << std::endl;
	std::cout << n[size-10] << std::endl;

	CHECK(cudaFree(gpu_Vm));

	// reset device
	CHECK(cudaDeviceReset());
}
