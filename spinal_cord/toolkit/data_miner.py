import os
from pkg_resources import resource_filename


class DataMiner:
    @staticmethod
    def get_average_voltage(neuron_group_name: str) -> dict:
        neurons = []
        average_data = dict()
        results_dir = resource_filename('spinal_cord', 'results')
        for result_file in os.listdir(results_dir):
            if neuron_group_name in result_file:
                with open(os.path.join(results_dir, result_file)) as datafile:
                    for line in datafile:
                        neuron_id, time, value = line.split()
                        if neuron_id not in neurons:
                            neurons.append(int(neuron_id))
                        if time not in average_data.keys():
                            average_data[float(time)] = 0.
                        average_data[float(time)] += float(value)
        for key in average_data.keys():
            average_data[key] = round(average_data[key] / len(neurons), 3)
        return average_data
