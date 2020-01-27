import numpy as np
import random
import math
import h5py
import subprocess
import logging
from tests_runner import convert_to_hdf5
from tests_runner import plot_results1
from ks_comparing import ks_data_test


class Individual:
    gen_length = 178  # weights and delays

    def __eq__(self, other):
        return self.pvalue == other.pvalue

    def __gt__(self, other):
        return self.pvalue > other.pvalue

    def __init__(self):
        self.pvalue = 0
        self.dvalue = 0
        self.gen = []
        self.test_result = []

    def __len__(self):
        return len(self.gen)


class Population:
    def __init__(self):
        self.population = []

    def add_individual(self, individual):
        self.population.append(individual)

    def __len__(self):
        return len(population.population)

    def format(self, x):
        return float("{0:.4f}".format(x))

    # init 10 individuals for first population
    def first_init(self):
        for i in range(10):
            individual = Individual()

            # EES, E1, E2, E3, E4, E5
            for j in range(0, 5):
                individual.gen.append(self.format(random.uniform(0.1, 250)))

            # CV 1-5 to iIP_E
            for j in range(5, 10):
                individual.gen.append(self.format(random.uniform(0.1, 50)))

            # iIP_E to eIP_E,F
            for j in range(10, 12):
                individual.gen.append(self.format(random.uniform(0.0001, 30)))

            # iIP_E to OM1~5F
            for j in range(12, 16):
                individual.gen.append(self.format(random.uniform(0.0001, 5)))

            # eIP_E,F to MN_E,F
            for j in range(16, 18):
                individual.gen.append(self.format(random.uniform(0.0001, 10)))

            for j in range(18, 32):
                individual.gen.append(self.format(random.uniform(0.0001, 20)))

            # init delays
            for j in range(32, 64):
                individual.gen.append(self.format(random.uniform(0.2, 10)))


            # # init weights
            # # (E1, E2, E3, E4, E5 -- OMs)
            # for j in range(0, 5):
            #     individual.gen.append(self.format(random.uniform(0.0001, 10)))
            #
            # # CVs -- OMs_0
            # for j in range(5, 14):
            #     individual.gen.append(self.format(random.uniform(0.0001, 1)))
            #
            # # CVs -- OMs_3
            # for j in range(14, 20):
            #     individual.gen.append(self.format(random.uniform(0.0001, 1)))
            #
            # # inner connectomes
            # for j in range(20, 89):
            #     individual.gen.append(self.format(random.uniform(0.0001, 20)))
            #
            # # init delays
            # for j in range(89, individual.gen_length):
            #     individual.gen.append(self.format(random.uniform(0.2, 10)))

            self.population.append(individual)


class Fitness:

    # calculate fitness function for instance of Invididual class
    @staticmethod
    def calculate_fitness(individual):
        individual.dvalue, individual.pvalue = ks_data_test(Data.path_to_bio_data, Data.path_to_test_data)

    # choose best value of fitness function for population
    @staticmethod
    def best_fitness(population):

        # best = -1
        # individ = Individual()
        # for individual in population.population:
        #     if individual.pvalue > best:
        #         best = individual.pvalue
        #         individ = individual

        # return individ
        return max(population.population)


class Breeding:

    @staticmethod
    def crossover(individual_1, individual_2):

        crossover_point = random.randint(0, len(individual_1))

        new_individual_1 = Individual()
        new_individual_1.gen = np.concatenate((np.array(individual_1.gen[0:crossover_point]),
                                         np.array(individual_2.gen[crossover_point:len(individual_1)])))
        new_individual_2 = Individual()
        new_individual_2.gen = np.concatenate((np.array(individual_2.gen[0:crossover_point]),
                                               np.array(individual_1.gen[crossover_point:len(individual_1)])))

        return new_individual_1, new_individual_2


    @staticmethod
    def mutation(individual):
        mutation_point = random.randint(0, len(individual))

        for index in range(mutation_point):

            mean = individual.gen[index]
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

            individual.gen[index] = Population().format(random.uniform(low, high))

    # return best 2 individuals from population
    @staticmethod
    def select(population):
        best_fitness = -1
        best_fitness_index = 0

        for i in range(len(population)):
            if population.population[i].pvalue > best_fitness:
                best_fitness = population.population[i].pvalue
                best_fitness_index = i

        best_fitness_2 = -1
        best_fitness_index_2 = 0

        for i in range(len(population)):
            if population.population[i].pvalue > best_fitness_2 and population.population[i].pvalue < best_fitness:
                best_fitness_2 = population.population[i].pvalue
                best_fitness_index_2 = i

        return population.population[best_fitness_index], population.population[best_fitness_index_2]

    # calculate test result for each individual in population
    # and write it in field test_result[]
    @staticmethod
    def calculate_tests_result(population, number):
        test = 1
        for individual in population.population:
            print("Test number: " + str(test))
            test += 1
            Build.run_test(individual, number)


class Evolution:

    @staticmethod
    def start_gen_algorithm(population, number):

        # calculate test result for population
        Breeding.calculate_tests_result(population, number)

        # select 2 best
        individual_1, individual_2 = Breeding.select(population)

        # create new population
        new_population = Population()

        # fill 10 individuals
        for i in range(5):
            crossover_individual_1, crossover_individual_2 = Breeding.crossover(individual_1, individual_2)
            new_population.add_individual(crossover_individual_1)
            new_population.add_individual(crossover_individual_2)

        # change 4 random of them
        for i in range(4):
            index = random.randint(0, len(population) - 1)
            Breeding.mutation(population.population[index])

        # add old best individuals
        # if new will be worst
        new_population.add_individual(individual_1)
        new_population.add_individual(individual_2)

        return new_population

    @staticmethod
    def mutation_for_population(population):
        for individual in population.population:
            Breeding.mutation(individual)

    @staticmethod
    def terminate(count):
        too_long = count > 1000

        solution_is_find = Evolution.solution()
        return too_long or solution_is_find

    @staticmethod
    def solution():
        return False


class Data:
    slice_number = 12
    ms_in_slice = 25
    sim_step_bio_data = 0.1
    sim_step_test_data = 0.025

    path_to_bio_data = "/home/yuliya/Desktop/genetic_algorithm/bio_data/foot" \
                       "/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5"

    path_to_test_data = "/home/yuliya/Desktop/genetic_algorithm/" \
                        "gras_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5"

    @staticmethod
    def read_bio_data():
        end = int(Data.ms_in_slice * Data.slice_number / Data.sim_step_bio_data)
        with h5py.File(Data.path_to_bio_data, 'r') as file:
            return np.array([data[:] for data in file.values()])[1][0:end]

    @staticmethod
    def read_test_data():
        end = int(Data.ms_in_slice * Data.slice_number / Data.sim_step_test_data)
        with open('MN_E.dat') as file:
            return list(map(float, file.read().split()))[0:end:4]


class Build:
    buildname = "build"
    script_place = "/home/yuliya/Desktop/genetic_algorithm"
    nvcc = "/usr/local/cuda-10.1/bin/nvcc"
    buildfile = "50_neurons.cu"

    @staticmethod
    def create_build_string(individual):
        return ' ' + ' '.join(map(str, individual.gen))

    @staticmethod
    def run_test(individual, number):
        logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
        logger = logging.getLogger()

        cmd_build = f"{Build.nvcc} -o {Build.script_place}/{Build.buildname} {Build.script_place}/{Build.buildfile}"
        cmd_run = f"{Build.script_place}/{Build.buildname}" + Build.create_build_string(individual)

        for cmd in [cmd_build, cmd_run]:
            logger.info(f"Execute: {cmd}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = process.communicate()

            for output in str(out.decode("UTF-8")).split("\n"):
                logger.info(output)
            for error in str(err.decode("UTF-8")).split("\n"):
                logger.info(error)


        convert_to_hdf5("/home/yuliya/Desktop/genetic_algorithm", number)

        individual.test_result = Data.read_test_data()

        f = open('pvalues.dat', 'a')

        Fitness.calculate_fitness(individual)
        f.write(f"Population {number}: individual pvalue = {individual.pvalue} dvalue: {individual.dvalue}\n")


class Debug:

   @staticmethod
   def save(population, number):

       individual = Fitness.best_fitness(population)

       folder = "/home/yuliya/Desktop/genetic_algorithm/"

       # print in file weights and result fitness function

       f = open('history.dat', 'a')
       f.write(f"Population number: {number}")
       f.write("\n2d \n")
       f.write(f"Dvalue = {individual.dvalue} \n")
       f.write(f"Pvalue = {individual.pvalue}" "\n")
       f.write("Weights: \n")
       for i in individual.gen:
           f.write(str(i) + " ")

       f.write("\n\n")

       plot_results1(folder, number)


if __name__ == "__main__":

    bio_data = Data.read_bio_data()

    # first initialization to population
    population = Population()
    population.first_init()

    i = 1
    population_number = 1

    while not Evolution.terminate(population_number):

        print("Population number = " + str(population_number))

        # if population_number % 10 == 0:
        #     Debug.save(population, population_number)
        #     i += 1

        new_population = Evolution.start_gen_algorithm(population, population_number)

        Debug.save(population, population_number)

        population = new_population

        population_number += 1

        f = open('history.dat', 'a')
        f.write("\n")

