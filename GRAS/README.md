# Neural topology based on Izhikevich neuron model implemented on GPU (CUDA)

## Description:
Real time simulation of neural topology for [Memristive spinal cord project](https://github.com/research-team/memristive-spinal-cord)

### Installation guide

For running a GRAS code you need to have NVIDIA video card and installed CUDA package (instruction for installing: 
https://developer.nvidia.com/cuda-toolkit). 
After packages preparations, run the command in a terminal:
```nvidia-smi``` 
If you have video card usage information then move further.
Download the stable version of the code from the repo (https://github.com/research-team/memristive-spinal-cord) and do any changes in the code if need (change topology, change output paths, thread numbers and etc.). 

Profiling can be made via next commands:
```nvprof ./build\_file```

More profiling  details: https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview

```cuda-memcheck ./build\_file```
More profiling details: https://docs.nvidia.com/cuda/cuda-memcheck/index.html


The best thread and block number which depends on your video card can be calculated here:
https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html

Open a terminal and enter the next command to compile a code via basic template: 
```nvcc -O3 -lcurand -o build /path/to/file```

- **nvcc** is the Nvidia CUDA Compiler 
- **-O3** flag is necessary for boosting the code
- **-lcurand** allows to use functions from the cuRAND library
- **build** is a filename of compiling result
- **/path/to/file** the path to the source file for compiling 

More details of compiling: 
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html

Then run:
```./build```

The code will be executed and you will get the results in a folder which you specify in the code
