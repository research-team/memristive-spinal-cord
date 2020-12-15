#include <iostream>

#define CHECK(call) {                                          \
    const cudaError_t error = call;                            \
    if (error != cudaSuccess) {                                  \
        fprintf(stderr, "Error: %s:%d\ncode: %d\n%s\n",        \
        __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(error);                                                 \
	}                                                               \
}
#include <vector>



struct States {
	float *Vm;
	float *n;
	float *m;
	unsigned int size;
};

struct Parameters {
	float *Cm;
	float *gnabar;
	float *E_ex;
	unsigned int size;
};



__global__
void some_kernel(States *s, int size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		s->Vm[id] += 500;
		s->m[id] -= 100;
		s->n[id] += 1.45;
	}
}

using namespace std;

int main(int argc, char **argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	const int size = 10000;
	// init structs
	size_t states_size = sizeof(States);
	States *dev_S, *S = (States *)malloc(states_size);

	// Allocate and fill host data
	float *dev_Vm, *Vm = new float[size]();
	float *dev_n, *n = new float[size]();
	float *dev_m, *m = new float[size]();

	for (int i = 0; i < size; i++) {
		Vm[i] = 500;
		m[i] = 200;
		n[i] = 10;
	}

	vector<void*> dev_VARS;
	dev_VARS.push_back(dev_Vm);
	cout << dev_Vm << endl;
	cout << &dev_Vm << endl;
	cout << dev_VARS[0] << endl;
	cout << **dev_VARS[0] << endl;
	exit(0);
	dev_VARS.push_back(&dev_n);
	dev_VARS.push_back(&dev_m);

	vector<void*> host_VARS;
	host_VARS.push_back(&Vm);
	host_VARS.push_back(&n);
	host_VARS.push_back(&m);
//
//	// Allocate device struct
//	cudaMalloc((States **) &dev_S, states_size);
//	// Allocate device array pointers
//	for (int i = 0; i < 3; i++){
//		CHECK(cudaMalloc((void **) &dev_VARS[i], size * sizeof(*dev_VARS[i])));
//	}
//
//	// Allocate device pointers
////	cudaMalloc((void **) &dev_Vm, size * sizeof(*dev_Vm));
////	cudaMalloc((void **) &dev_n, size * sizeof(*dev_n));
////	cudaMalloc((void **) &dev_m, size * sizeof(*dev_m));
//	// Copy pointer content from host to device.
//	for (int i = 0; i < 3; ++i) {
//		CHECK(cudaMemcpy(dev_VARS[i], host_VARS[i], size * sizeof(*host_VARS[i]), cudaMemcpyHostToDevice));
////		cudaMemcpy(dev_Vm, Vm, size * sizeof(*Vm), cudaMemcpyHostToDevice);
//	}
//
//	// Point to device pointer in host struct
//	S->Vm = dev_Vm;
//	S->n = dev_n;
//	S->m = dev_m;
//	// Copy struct from host to device.
//	cudaMemcpy(dev_S, S, sizeof(States), cudaMemcpyHostToDevice);
//
//	// Call kernel
//	some_kernel<<<10000/256 + 1, 256>>>(dev_S, size); // block size need to be a multiply of 256
//	CHECK(cudaDeviceSynchronize());
//
//	// Copy result to host:
//	cudaMemcpy(Vm, dev_Vm, size * sizeof(*Vm), cudaMemcpyDeviceToHost);
//	cudaMemcpy(m, dev_m, size * sizeof(*m), cudaMemcpyDeviceToHost);
//	cudaMemcpy(n, dev_n, size * sizeof(*n), cudaMemcpyDeviceToHost);
//
//	// Print some result
//	std::cout << Vm[size-10] << std::endl;
//	std::cout << m[size-10] << std::endl;
//	std::cout << n[size-10] << std::endl;
//
//	CHECK(cudaFree(dev_Vm));
//
//	// reset device
//	CHECK(cudaDeviceReset());

//	std::cout << host_arr2[size-1] << std::endl;
//	std::cout << host_arr3[size-1] << std::endl;
}