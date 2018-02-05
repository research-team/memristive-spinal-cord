import os
from collections import OrderedDict


class DeviceData:
    """
    format of _items: dict(<simulation_time>: dict(<neuron_id>: <list of recorder values>))
    """

    def __init__(self, data_name_list: list) -> None:
        self._items = OrderedDict()
        self._data_name_list = data_name_list

    def add_item(self, time: float, neuron_id: int, value_list: list) -> None:
        if not self.has_item(time):
            self.create_item(time)
        item = self.get_item(time)
        if str(neuron_id) in item:
            raise ValueError("DeviceData data item " + str(time) + " already has neuron_id " + str(neuron_id))
        item[str(neuron_id)] = value_list

    def has_item(self, time) -> bool:
        return time in self._items

    def get_item(self, time) -> float:
        return self._items[time]

    def create_item(self, time) -> None:
        self._items[time] = dict()

    def get_data(self) -> OrderedDict:
        return self._items

    def get_data_for_single_neuron(self, neuron_id: int, items) -> OrderedDict:
        return self.get_data_for_neurons([neuron_id], items)

    def get_data_for_neurons(self, neuron_id_list: list, items=None) -> OrderedDict:
        result = OrderedDict()
        if items is None:
            items = self._items
        for time, item in items.items():
            item_for_neurons = {neuron_id: values for neuron_id, values in item.items() if neuron_id in neuron_id_list}
            result[time] = item_for_neurons
        return result

    def filter_items_for_value(self, value_name: str, items=None) -> OrderedDict:
        result = OrderedDict()
        data_index = self._data_name_list.index(value_name)
        if data_index is not None:
            if items is None:
                items = self._items
            for time, item in items.items():
                item = {neuron_id: values[data_index] for neuron_id, values in item.items()}
                result[time] = item
        return result

    def average(self, value_name: str) -> OrderedDict:
        result = OrderedDict()
        for time, item in self.filter_items_for_value(value_name).items():
            values_list = item.values()
            average_value = round(sum(values_list) / float(len(values_list)), 3)
            result[time] = average_value
        return result


def extract_device_data(device_name, value_names, storage_dir: str) -> DeviceData or None:
    device_filepath = None
    for file in os.listdir(storage_dir):
        if file.startswith(device_name.value):
            device_filepath = storage_dir + '/' + file
            break

    if device_filepath is not None:
        data = DeviceData(value_names)
        with open(device_filepath) as device_file:
            for line in device_file:
                line = line.split()
                time = float(line[1])
                neuron_id = int(line[0])
                values = [float(value) for value in line[2:]]
                data.add_item(time, neuron_id, values)
        return data
    return None


def get_average_voltage(device_name, storage_dir):
    data = extract_device_data(
        device_name,
        ['V_m'],
        storage_dir
    )
    return data.average('V_m')
