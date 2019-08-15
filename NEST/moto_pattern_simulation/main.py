import nest
import nestData
import multimetersDef
import newNest
import plot

HZ = 40.0  # герц

nest.ResetKernel()
nest.SetKernelStatus({
    'total_num_virtual_procs': 4,
    'print_time': True,
    'resolution': 0.025,
    'overwrite_files': True})

all_neurons = newNest.create_network()

multimeters = multimetersDef.multimeters_creater(nest, all_neurons)
spike_detectors = multimetersDef.spike_det_creater(nest, all_neurons)

multimetersDef.multimeters_connector(nest, multimeters, all_neurons)
multimetersDef.spike_det_connector(nest, spike_detectors, all_neurons)

nest.Simulate(nestData.SIM_TIME)

plot.plot_pics(all_neurons, multimeters=multimeters, spike_detectors=spike_detectors)
