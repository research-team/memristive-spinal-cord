import numpy as np
import random
from scipy.stats import ks_2samp
import csv
import math
import h5py
import subprocess

class Individual:
    gen_length = 43 * 2  # weigths and delays

    def __init__(self):
        self.fitness = -1
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

    # init 10 individuals for frst population
    def first_init(self):
        for i in range(10):
            individual = Individual()
            # init weights
            for j in range(0, 16):
                individual.gen.append(random.uniform(0.00000001, 10))
            # init weights
            for j in range(16, int(individual.gen_length / 2)):
                individual.gen.append(random.uniform(0.00000001, 1))
            # init delays
            for j in range(int(individual.gen_length / 2), individual.gen_length):
                individual.gen.append(random.uniform(0.2, 10))
            self.population.append(individual)

class Fitness:

    # calculate fitness function for instance of Invididual class
    @staticmethod
    def calculate_fitness(individual, bio_data):
        individual.fitness = ks_2samp(individual.test_result, bio_data)[1]


    # calculate fitness function for each individual in population
    @staticmethod
    def calculate_fitness_for_population(population, bio_data):
        for individual in population.population:
            Fitness.calculate_fitness(individual, bio_data)

    # choose best value of fintness function for population
    @staticmethod
    def best_fitness(population):
        best = -1
        for individual in population.population:
            if individual.fitness > best:
                best = individual.fitness

        return best


class Breeding:

    @staticmethod
    def crossover(individual_1, individual_2):
        crossover_point = int(random.uniform(0, len(individual_1)))
        new_individual_1 = Individual()
        new_individual_1.gen = np.concatenate((np.array(individual_1.gen[0:crossover_point]),
                                         np.array(individual_2.gen[crossover_point:len(individual_1)])))
        new_individual_2 = Individual()
        new_individual_2.gen = np.concatenate((np.array(individual_2.gen[0:crossover_point]),
                                               np.array(individual_1.gen[crossover_point:len(individual_1)])))

        return new_individual_1, new_individual_2


    @staticmethod
    def mutation(individual):
        mutation_point = random.uniform(0, len(individual))
        mean = individual.gen[i]
        sigma = abs(mean) / 5
        probability = 0.001
        k = math.log(math.sqrt(2 * math.pi * probability * probability * sigma * sigma))
        res = sigma * math.sqrt(-2 * k) if k < 0 else sigma * math.sqrt(2 * k)
        low = mean - res
        high = mean + res
        for index in range(mutation_point):
            individual.gen[index] = random.uniform(low, high)

    # return best 2 individuals from population
    @staticmethod
    def select(population):
        best_fitness = -1
        best_fitness_index = 0

        for i in range(len(population)):
            if population.population[i].fitness > best_fitness:
                best_fitness = population.population[i].fitness
                best_fitness_index = i

        best_fitness_2 = -1
        best_fitness_index_2 = 0

        for i in range(len(population)):
            if population.population[i].fitness > best_fitness_2 and population.population[i].fitness < best_fitness:
                best_fitness_2 = population.population[i]
                best_fitness_index_2 = i

        return population.population[best_fitness_index], population.population[best_fitness_index_2]

    # calculate test result for each indbvidual in population
    # and write it in field test_esult[]
    @staticmethod
    def calculate_tests_result(population):
        for individual in population.population:
            Build.run_test(individual)

class Evolution:
    count = 0

    @staticmethod
    def start_gen_algorithm(population):
        individual_1, individual_2 = Breeding.select(population)
        new_population = Population()
        for i in range(5):
            crossover_individual_1, crossover_individual_2 = Breeding.crossover(individual_1, individual_2)
            new_population.add_individual(crossover_individual_1)
            new_population.add_individual(crossover_individual_2)
        return new_population

    @staticmethod
    def mutation_for_population(population):
        for individual in population.population:
            Breeding.mutation(individual)

    @staticmethod
    def terminate():
        too_long = count > 1000
        solution_is_find = solution()
        return too_long or solution_is_find

    @staticmethod
    def solution():
        return False


class Data:

    @staticmethod
    def read_bio_data(path):
        with h5py.File(path, 'r') as file:
            bio_data = np.array([data[:] for data in file.values()])
        return bio_data[1][0:500]

    @staticmethod
    def read_test_data():
        number_slices = 2
        end = int(25 * number_slices / 0.025)
        with open('MN_E.dat') as file:
            results_arr = list(map(float, file.read().split()))[0:end:4]
        return results_arr

class Build:

    @staticmethod
    def create_build_string(individual):
        return ' ' + ' '.join(map(str, individual.gen))

    @staticmethod
    def run_test(individual):
        buildname = "build"
        s = Build.create_build_string(individual)
        script_place = "/home/yuliya/Desktop/genetic_algorithm"
        nvcc = "/usr/local/cuda-10.1/bin/nvcc"
        buildfile = "two_muscle_simulation.cu"

        cmd_build = f"{nvcc} -o build {script_place}/{buildfile}"
        cmd_run = f"{script_place}/{buildname}" + s

        for cmd in [cmd_build, cmd_run]:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            process.communicate()

        individual.test_result = Data.read_test_data()


if __name__ == "__main__":
    # save bio_data results
    path_to_bio_data = "/home/yuliya/Desktop/genetic_algorithm/bio_data/foot/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5"
    bio_data = Data.read_bio_data(path_to_bio_data)

    # first initialization to population
    population = Population()
    population.first_init()

    # calculate first tests result and save it in individual.test_result
    Breeding.calculate_tests_result(population)

    # calculate first fitness function
    Fitness.calculate_fitness_for_population(population, bio_data)

    # save best value of fitness function on this step
    best_for_init_population = Fitness.best_fitness(population)

    # init new population
    new_population = Evolution.start_gen_algorithm(population)
    # calculate best fitness function for new population
    best_for_new_population = Fitness.best_fitness(new_population)

    if best_for_init_population > best_for_new_population:
        while best_for_init_population > best_for_init_population:
            Evolution.mutation_for_population(new_population)
            Breeding.calculate_tests_result(new_population)
            best_for_new_population = Fitness.best_fitness(new_population)
    best_for_init_population = best_for_new_population