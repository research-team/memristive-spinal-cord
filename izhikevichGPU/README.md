# Neural topology based on Izhikevich neuron model implemented on GPU (OpenACC)

### Description:
Real time simulation of neural topology for [Memristive spinal cord project](https://github.com/research-team/memristive-spinal-cord)

### Installation
1. Install the PGI
2. . . .



### Instruction "How to run the code"
You should have installed PGI and Nvidia videocard

1. Clone the **[git project](https://github.com/research-team/memristive-spinal-cord)**
2. Open **IzhikevichGPU** folder
3. Compile by the command:
```bash
pgc++ -acc -ta=tesla:managed,lineinfo -Minfo=all,intensity,ccff -o output main.cpp
```
4. Then run the program:
```bash
./output
```
5. Profiling by:
```bash
nvprof ./output
```
