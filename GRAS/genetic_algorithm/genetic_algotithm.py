import random
import math
import os
import logging
import numpy as np
import h5py as hdf5
import time
from multi_gpu_build import Build
from meta_plotting import get_4_pvalue

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

N_individuals_in_first_init = 1000
N_how_much_we_choose = 15
N_individuals_for_mutation = 15

# p-value what we want or bigger
p = 0.05

# connectomes number
N = 104

max_weights = []
low_weights = []

low_delay = 0.5
max_delay = 6

# TODO it's needed to be bigger or smaller ?
low_weight = 0.0001
max_weight = 2

# TODO it's needed now ?
# if number of spikes less then this then p-value = 0
critial_num_spikes = 10000

path = '/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test'


def convert_to_hdf5(result_folder):
    """
    Converts dat files into hdf5 with compression
    Args:
        result_folder (str): folder where is the dat files placed
    """
    # process only files with these muscle names
    for muscle in ["MN_E", "MN_F"]:
        logger.info(f"converting {muscle} dat files to hdf5")
        is_datfile = lambda f: f.endswith(f"{muscle}.dat")
        datfiles = filter(is_datfile, os.listdir(result_folder))

        name = f"gras_{muscle.replace('MN_', '')}_PLT_21cms_40Hz_2pedal_0.025step.hdf5"

        with hdf5.File(f"{result_folder}/{name}", 'w') as hdf5_file:
            for test_index, filename in enumerate(datfiles):
                with open(f"{result_folder}/{filename}") as datfile:
                    data = [float(v) for v in datfile.readline().split()]
                    # check on NaN values (!important)
                    if any(map(np.isnan, data)):
                        logging.info(f"{filename} has NaN... skip")
                        f = open(f"{result_folder}/time.txt", 'w')
                        f.write("0")
                        f.close()
                        f = open(f"{result_folder}/ampl.txt", 'w')
                        f.write("0")
                        f.close()
                        f = open(f"{result_folder}/2d.txt", 'w')
                        f.write("0")
                        f.close()
                        continue

                    length = len(data)
                    start, end, l = 0, 0, int(length / 10)
                    for i in range(10):
                        end += l
                        arr = data[start:end]
                        start += l
                        hdf5_file.create_dataset(f"{i}", data=arr, compression="gzip")
        # check that hdf5 file was written properly
        with hdf5.File(f"{result_folder}/{name}") as hdf5_file:
            assert all(map(len, hdf5_file.values()))


class Individual:

    def __str__(self):
        return f"p-value = {self.pvalue}, p-value ampls = {self.pvalue_amplitude}, p-value times = {self.pvalue_times}\n"

    def __eq__(self, other):
        return self.pvalue == other.pvalue

    def __gt__(self, other):
        return self.pvalue > other.pvalue

    def __init__(self):
        self.pvalue = 0.0
        self.pvalue_amplitude = 0.0
        self.pvalue_times = 0.0
        self.gen = []
        self.weights = []
        self.delays = []

    def __copy__(self):

        new_individual = Individual()

        for this in self.gen:
            new_individual.gen.append(this)

        return new_individual

    def __len__(self):
        return len(self.gen)

    def format(self, x):
        return float("{0:.2f}".format(x))

    def is_correct(self):
        return self.pvalue_times != 0 and self.pvalue_amplitude != 0 and self.pvalue != 0

    def init(self):

        # Es ~ OMs
        for i in range(5):
            self.weights.append(random.uniform(0.01, 0.3))
            low_weights.append(0.01)
            max_weights.append(0.3)

        # CVs - OMs
        for i in range(16):
            self.weights.append(random.uniform(0.01, 3))
            low_weights.append(0.01)
            max_weights.append(3)

        # output to Flexor another OM
        for i in range(4):
            self.weights.append(random.uniform(0.01, 5))
            low_weights.append(0.01)
            max_weights.append(5)

        # output to eIP
        for i in range(10):
            self.weights.append(random.uniform(0.1, 2.5))
            low_weights.append(0.1)
            max_weights.append(2.5)

        # TODO init ex weights
        for i in range(40):
            self.weights.append(random.uniform(0.05, 1.2))
            low_weights.append(0.05)
            max_weights.append(1.2)

        # TODO init inh weights
        for i in range(15):
            self.weights.append(random.uniform(0.01, 0.2))
            low_weights.append(0.01)
            max_weights.append(0.2)

        for i in range(2):
            self.weights.append(random.uniform(0.005, 0.06))
            low_weights.append(0.005)
            max_weights.append(0.06)

        for i in range(4):
            self.weights.append(random.uniform(0.0005, 0.002))
            low_weights.append(0.0005)
            max_weights.append(0.002)

        # eIP ~ MN
        for i in range(2):
            self.weights.append(random.uniform(5, 10))
            low_weights.append(5)
            max_weights.append(10)

        for i in range(6):
            self.weights.append(0)
            low_weights.append(0)
            max_weights.append(0)

        # init delays
        for i in range(N):
            self.delays.append(random.uniform(low_delay, max_delay))

        self.gen = self.weights + self.delays


class Population:

    def __init__(self):
        self.individuals = []

    def add_individual(self, individual):
        self.individuals.append(individual)

    def __len__(self):
        return len(self.individuals)

    # init N_individuals_in_first_init individuals for first population
    def first_init(self):
        for i in range(N_individuals_in_first_init):
            individual = Individual()
            individual.init()
            self.individuals.append(individual)

    # TODO first init for knowing part of weights and delays, it's needed ?


class Fitness:

    # calculate fitness function for instance of Invididual class
    @staticmethod
    def calculate_fitness(individuals, num_population):

        # converting 4 results data to hdf5
        for i in range(4):
            convert_to_hdf5(f"{path}/dat/{i}")

        # get p-value for 4 individuals
        get_4_pvalue()

        # set p-value to this individuals
        for i in range(len(individuals)):
            individual = individuals[i]

            ampl = open(f'{path}/dat/{i}/a.txt')
            times = open(f'{path}/dat/{i}/t.txt')
            d2 = open(f'{path}/dat/{i}/d2.txt')

            individual.pvalue_amplitude = float(ampl.readline())
            individual.pvalue_times = float(times.readline())
            individual.pvalue = float(d2.readline())

            Fitness.write_pvalue(individual, num_population)

    @staticmethod
    def write_pvalue(individual, number):

        f = open(f'{path}/files/history.dat', 'a')

        if individual.pvalue != 0:
            f.write(f"Population {number}: individual pvalue: {individual.pvalue}\n")
            f.write(f"individual pvalue_ampl: {individual.pvalue_amplitude}\n")
            f.write(f"individual pvalue_time: {individual.pvalue_times}\n")
            f.write(f"Weighs and delays: {' '.join(map(str, individual.gen))}\n")
        else:
            f.write(f"Population {number}: individual pvalue: {individual.pvalue}\n\n")

        f.close()

    # choose best value of fitness function for population
    @staticmethod
    def best_fitness(current_population):
        return max(current_population.individuals)


class Breeding:

    @staticmethod
    def crossover(individual_1, individual_2):
        length = len(individual_1)

        crossover_point = random.randint(0, length)

        new_individual_1 = Individual()
        new_individual_1.gen = individual_1.gen[0:crossover_point] + individual_2.gen[crossover_point:length]

        new_individual_2 = Individual()
        new_individual_2.gen = individual_2.gen[0:crossover_point] + individual_1.gen[crossover_point:length]

        return new_individual_1, new_individual_2

    @staticmethod
    def mutation2(individual):

        new_individual = individual.__copy__()

        n = random.randint(0, 100)

        for index in range(len(individual)):
            if n % 2 == 0:
                if index % 2 == 0:
                    mean = new_individual.gen[index]
                    low, high = Breeding.get_low_high(mean)
                    if index < 5:
                        new_individual.gen[index] = random.randint(int(low), int(high))
                    else:
                        new_individual.gen[index] = Individual().format(random.uniform(low, high))
            else:
                if index % 2 != 0:
                    mean = new_individual.gen[index]
                    low, high = Breeding.get_low_high(mean)
                    if index < 5:
                        new_individual.gen[index] = random.randint(int(low), int(high))
                    else:
                        new_individual.gen[index] = Individual().format(random.uniform(low, high))

        return new_individual

    @staticmethod
    def get_low_high(mean):

        mean = float(mean)

        sigma = abs(mean) / 5
        probability = 0.001
        n = math.sqrt(2 * math.pi * probability * probability * sigma * sigma)

        if n == 0:
            n = probability

        k = math.log(n)
        res = sigma * math.sqrt(-2 * k) if k < 0 else sigma * math.sqrt(2 * k)
        low = mean - res

        if low < 0:
            low = probability / 10

        high = mean + res

        return low, high

    # TODO add another way to get low and high

    @staticmethod
    def mutation3(individual):

        new_individual = individual.__copy__()

        for index in range(len(individual)):
            n = random.randint(0, 100)
            if n < 50:
                mean = new_individual.gen[index]
                low, high = Breeding.get_low_high(mean)
                if index < 5:
                    new_individual.gen[index] = random.randint(int(low), int(high))
                else:
                    new_individual.gen[index] = Individual().format(random.uniform(low, high))

        return new_individual

    @staticmethod
    def mutation4(individual):

        new_individual = individual.__copy__()

        for index in range(len(individual)):
            n = random.randint(0, 100)
            if n < 50:
                m = random.randint(2, 10)
                mean = new_individual.gen[index]
                low, high = mean - mean / m, mean + mean / m
                if index < 5:
                    new_individual.gen[index] = random.randint(int(low), int(high))
                else:
                    new_individual.gen[index] = Individual().format(random.uniform(low, high))

        return new_individual

    @staticmethod
    def mutation(individual):

        new_individual = individual.__copy__()
        mutation_point = random.randint(0, len(individual))

        for index in range(mutation_point):
            mean = new_individual.gen[index]
            low, high = Breeding.get_low_high(mean)
            if index < 5:
                new_individual.gen[index] = random.randint(int(low), int(high))
            else:
                new_individual.gen[index] = Individual().format(random.uniform(low, high))

        return new_individual

    # return best N_how_much_we_choose individuals from population
    @staticmethod
    def select(current_population):

        len_current_population = len(current_population)
        logg_string = f"Length current population = {len_current_population}\n"

        newPopulation = Population()

        counter = 0

        # skip incorrect individuals
        for index in range(len_current_population):

            if current_population.individuals[index].is_correct():
                newPopulation.add_individual(current_population.individuals[index])

            else:
                counter += 1
                s = f"Skip individual because p-value = {current_population.individuals[index].pvalue} "
                s += f"pvalue_ampl = {current_population.individuals[index].pvalue_amplitude} "
                s += f"pvalue_ampl = {current_population.individuals[index].pvalue_times}\n"
                print(s)
                logg_string += s

        logg_string += f"Skiped {counter} individuals\n"
        file = open(f"{path}/files/log.dat", 'a')
        file.write(logg_string)
        file.close()

        # sort this individuals
        arr = sorted(newPopulation.individuals, reverse=True)

        if len(arr) > N_how_much_we_choose:
            return arr[0:N_how_much_we_choose]
        else:
            return arr

    @staticmethod
    def calculate_tests_result(current_population, number):

        arr1 = []
        l = len(current_population)
        b = int(l / 4)
        cp = current_population[0:b * 4]
        k = 0

        while True:
            arr = []
            for i in range(4):
                arr.append(cp[k])
                k += 1
            arr1.append(arr)
            if k >= len(cp):
                break

        arr1.append(current_population[b * 4:l])

        for i in range(len(arr1)):
            Build.run_tests(arr1[i])
            time.sleep(0.1)
            Fitness.calculate_fitness(arr1[i], number)


class Evolution:
    terminate_algorithm = False

    @staticmethod
    def start_gen_algorithm(current_population, number):

        # Evolution.terminate_algorithm = max(current_population.individuals).pvalue >= p

        Breeding.calculate_tests_result(current_population.individuals, number)

        file_with_log_of_bests = open(f'{path}/files/log_of_bests.dat', 'a')

        individuals = Breeding.select(current_population)
        best_individual = max(individuals)
        file = open('../files/bests_pvalue.dat', 'a')
        file.write(f"In population number {number} best pvalue = {best_individual.pvalue} \n")
        file.write(f"pvalue_ampl = {best_individual.pvalue_amplitude} \n pvalue_times = {best_individual.pvalue_times}\n")
        file.write(' '.join(map(str, best_individual.gen)) + "\n")

        file_with_log_of_bests.write(f"{''.join(map(str, individuals))}\nWas chosen {best_individual}\n\n")

        newPopulation = Population()

        length_population = len(individuals)

        # each with each
        for i in range(length_population):
            individual_1 = individuals[i]
            j = i
            while j + 1 < length_population:
                j += 1
                individual_2 = individuals[j]
                crossover_individual_1, crossover_individual_2 = Breeding.crossover(individual_1, individual_2)
                newPopulation.add_individual(crossover_individual_1)
                newPopulation.add_individual(crossover_individual_2)

        print("Len population after crossover: " + str(len(newPopulation)))
        f = open(f'{path}/files/len_population.dat', 'a')
        f.write(f"Len population after crossover: {len(newPopulation)} ")

        for individual in newPopulation.individuals:
            for i in range(len(individual.gen)):
                individual.gen[i] = float(individual.gen[i])

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation3(individual))

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation2(individual))

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation4(individual))

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation(individual))

        for individual in individuals:
            if individual.is_correct():
                newPopulation.add_individual(individual)

        final_population = Population()

        for j in range(len(newPopulation)):

            individual = newPopulation.individuals[j]

            for i in range(len(individual)):

                if i >= int(len(individual) / 2):
                    if float(individual.gen[i]) < low_delay:
                        individual.gen[i] = low_delay
                    if float(individual.gen[i]) > max_delay:
                        individual.gen[i] = max_delay

                if i < int(len(individual) / 2):
                    if float(individual.gen[i]) < low_weights[i]:
                        individual.gen[i] = low_weights[i]
                    if float(individual.gen[i]) > max_weights[i]:
                        individual.gen[i] = max_weights[i]

            final_population.add_individual(individual)

        f.write(f"Final len: {len(final_population)} ")

        return final_population


class Debug:

   @staticmethod
   def save(current_population, number):

       current_individual = Fitness.best_fitness(current_population)

       folder = f"{path}/files/"
       
       # print in file weights and result fitness function
       f = open(f'{folder}/history.dat', 'a')
       f.write(f"Population number: {number}")
       f.write(f"\nPvalue_time = {current_individual.pvalue_times}")
       f.write(f"\nPvalue_ampl = {current_individual.pvalue_amplitude}")
       f.write("\n2d\n")
       f.write(f"Pvalue = {current_individual.pvalue}" "\n")
       f.write("Weights and delays: \n")
       for i in current_individual.gen:
           f.write(str(i) + " ")

       f.write("\n\n")


if __name__ == "__main__":

    files = [f"{path}/files/history.dat", f"{path}/files/bests_pvalue.dat", f"{path}/files/log.dat"]

    for file in files:
        if os.path.isfile(file):
            os.remove(f"{file}")

    print("Deleted files with history, bests_pvalue and log")

    Build.compile()

    # first initialization to population
    population = Population()
    population.first_init()

    i = 1
    population_number = 1

    while not Evolution.terminate_algorithm:

        print(f"Population number = {population_number}")

        new_population = Evolution.start_gen_algorithm(population, population_number)

        Debug.save(population, population_number)

        population = new_population

        population_number += 1