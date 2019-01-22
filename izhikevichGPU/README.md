# Neural topology based on Izhikevich neuron model implemented on GPU (CUDA)

### Description:
Real time simulation of neural topology for [Memristive spinal cord project](https://github.com/research-team/memristive-spinal-cord)

### Installation
1. Install the CUDA

### Instruction "How to run the code"
You should have installed CUDA and Nvidia videocard

1. Clone the **[git project](https://github.com/research-team/memristive-spinal-cord)**
2. Open **IzhikevichGPU** folder
3. Compile by the command:
```bash
nvcc -lineinfo -o output cuda_sim.cu
```
4. Then run the program:
```bash
./output
```
5. Profiling by:
```bash
nvprof ./output
```
### Technical description (in progress):
*Threads in a block* = 32 (warp size)  
*Number of blocks* = number of synapses / 32 + 1

*Total number of threads* = *Number of blocks* * *Threads in a block*. It is a little bit more than number of synapses, so extra threads are not used in simulation (number of extra threads is not bigger than *Threads in a block*)
```c++
for(int iter_step = 0; iter_step < ms_to_step(T_sim); iter_step++)
  GPU_kernel<<<. . .>>>(. . .);
```


![GPU](doc/GPU.png)
