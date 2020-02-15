import random
import math
import logging
from python_scripts.converter import convert_to_hdf5
from python_scripts.ks_comparing import ks_data_test
from python_scripts.gcc_build import Build

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

class Data:

    path_to_bio_data = "/home/yuliya/Desktop/ga_v3/data/reflex_arc/gras_E_15cms_40Hz_i100_2pedal_no5ht_T_0.25step.hdf5"

    path_to_test_data = "/home/yuliya/Desktop/ga_v3/dat/gras_E_15cms_40Hz_i100_2pedal_no5ht_T_0.25step.hdf5"

    # path_to_bio_data = "/home/yuliya/Desktop/genetic_algorithm/bio_data/foot" \
    #                    "/bio_E_21cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5"
    #
    # path_to_test_data = "/home/yuliya/Desktop/genetic_algorithm/" \
    #                     "gras_E_21cms_40Hz_i100_2pedal_no5ht_T_0.025step.hdf5"

class Individual:

    def __eq__(self, other):
        return self.pvalue == other.pvalue

    def __gt__(self, other):
        return self.pvalue > other.pvalue

    def __init__(self):
        self.pvalue = 0
        self.dvalue = 100
        self.gen = []
        self.weights = []
        self.delays = []
        self.dvalue_amplitude = 0
        self.pvalue_amplitude = 0
        self.dvalue_times = 0
        self.pvalue_times = 0

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
        return float("{0:.1f}".format(x))

    # init 200 individuals for first population
    def first_init(self):
        for i in range(200):
            individual = Individual()

            # init weights
            for j in range(0, 32):
                individual.weights.append(random.randint(0, 10000))

            # init delays
            for j in range(32, 64):
                individual.delays.append(self.format(random.uniform(0.5, 6)))

            individual.gen = individual.weights + individual.delays

            self.population.append(individual)


class Fitness:

    # calculate fitness function for instance of Invididual class
    @staticmethod
    def calculate_fitness(individual):
        individual.dvalue, individual.pvalue,\
            individual.dvalue_times, individual.pvalue_times,\
            individual.dvalue_amplitude, individual.pvalue_amplitude \
            = ks_data_test(Data.path_to_bio_data, Data.path_to_test_data)

    # choose best value of fitness function for population
    @staticmethod
    def best_fitness(population):
        return max(population.population)


class Breeding:

    @staticmethod
    def crossover(individual_1, individual_2):
        length = len(individual_1)

        crossover_point = random.randint(0, len(individual_1))

        new_individual_1 = Individual()
        new_individual_1.gen = individual_1.gen[0:crossover_point] + individual_2.gen[crossover_point:length]

        new_individual_2 = Individual()
        new_individual_2.gen = individual_2.gen[0:crossover_point] + individual_1.gen[crossover_point:length]

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

    # return best 10 individuals from population
    @staticmethod
    def select(population):
        arr = sorted(population)[0:10]
        return arr

    # calculate test result for each individual in population
    # and write it in field test_result[]
    @staticmethod
    def calculate_tests_result(population, number):
        test = 1
        for individual in population.population:
            print("Test number: " + str(test))
            test += 1
            Build.build(individual)
            Breeding.save_fitness(individual, number)

    @staticmethod
    def save_fitness(individual, number):
        convert_to_hdf5("/home/yuliya/Desktop/genetic_algorithm", number)

        f = open('dvalues.dat', 'a')

        Fitness.calculate_fitness(individual)
        f.write(f"Population {number}: individual dvalue = {individual.dvalue} pvalue: {individual.pvalue}\n")


class Evolution:
    terminate_algorithm = False

    @staticmethod
    def start_gen_algorithm(population, number):
        Breeding.calculate_tests_result(population, number)

        individuals = Breeding.select(population)

        new_population = Population()

        length_population = len(population)

        # each with each
        for i in range(length_population):
            individual_1 = individuals[i]
            j = i
            while j + 1 < length_population:
                j += 1
                individual_2 = individuals[j]

                crossover_individual_1, crossover_individual_2 = Breeding.crossover(individual_1, individual_2)
                new_population.add_individual(crossover_individual_1)
                new_population.add_individual(crossover_individual_2)

        # change 10 random of them
        for i in range(10):
            index = random.randint(0, len(population) - 1)
            Breeding.mutation(population.population[index])

        # add old best individuals
        # if new will be worst
        new_population.add_individual(individuals[0])
        new_population.add_individual(individuals[1])

        Evolution.terminate_algorithm = max(population).pvalue >= 0.1

        return new_population

    @staticmethod
    def mutation_for_population(population):
        for individual in population.population:
            Breeding.mutation(individual)

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

       # plot_results1(folder, number)


if __name__ == "__main__":

    Build.compile()

    # first initialization to population
    population = Population()
    population.first_init()

    # raise Exception

    i = 1
    population_number = 1

    while not Evolution.terminate_algorithm:

        print("Population number = " + str(population_number))

        new_population = Evolution.start_gen_algorithm(population, population_number)

        Debug.save(population, population_number)

        population = new_population

        population_number += 1

        f = open('history.dat', 'a')
        f.write("\n")

