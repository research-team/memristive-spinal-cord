import json
from python_scripts.genetic_algotithm import Individual

if __name__ == "__main__":

    with open("../logs/log_of_bests.dat") as file:
        all_lines = file.readlines()

    arr = []

    for line in all_lines:
        arr.append(json.loads(line, object_hook=Individual.decode))

    individuals = arr[len(arr) - 1]

    f = open("known.dat", "w")

    for individual in individuals:
        f.write(f"{json.dumps(individual, default=Individual.encode)}\n")