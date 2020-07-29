import random
import math
import os
import logging
import numpy as np
import h5py as hdf5
import time
import json

from multi_gpu_build import Build
from meta_plotting import get_4_pvalue
from data import DataSettings

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

N_how_much_we_choose = DataSettings.N_choose

# p-value what we want or bigger
p = 0.05

# connectomes number
N = DataSettings.N_connectomes

speed = DataSettings.speed

max_weights = []
low_weights = []

low_delay = 0.2
max_delay = 6

path = '/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test'


# path = '/home/yuliya/Desktop/ga_v9'


def write_zero(folder, filename, extension="txt"):
    file_to_write_zero = open(f"{folder}/{filename}.{extension}", "w")
    file_to_write_zero.write("0")
    file_to_write_zero.close()


def write_zeros(result_folder):
    write_zero(result_folder, "time")
    write_zero(result_folder, "ampl")
    write_zero(result_folder, "2d")
    write_zero(result_folder, "dt")


def convert_to_hdf5(result_folder, tests=False):
    """
    Converts dat files into hdf5 with compression
    Args:
        result_folder (str): folder where is the dat files placed
        :param result_folder:
        :param tests:
    """
    # process only files with these muscle names
    for muscle in ["MN_E", "MN_F"]:
        logger.info(f"converting {muscle} dat files to hdf5")
        is_datfile = lambda f: f.endswith(f"{muscle}.dat")
        datfiles = filter(is_datfile, os.listdir(result_folder))

        name = f"gras_{muscle.replace('MN_', '')}_PLT_{speed}cms_40Hz_2pedal_0.025step.hdf5"

        with hdf5.File(f"{result_folder}/{name}", 'w') as hdf5_file:
            for test_index, filename in enumerate(datfiles):
                with open(f"{result_folder}/{filename}") as datfile:
                    try:

                        data = [-float(v) for v in datfile.readline().split()]

                        # check on NaN values (!important)
                        if any(map(np.isnan, data)):
                            logging.info(f"{filename} has NaN... skip")
                            write_zeros(result_folder)
                            continue

                        length = len(data)
                        start, end, l = 0, 0, int(length / DataSettings.steps)
                        for i in range(DataSettings.steps):
                            end += l
                            arr = data[start:end]
                            start += l
                            hdf5_file.create_dataset(
                                f"#1_112409_PRE_BIPEDAL_normal_21cms_burst7_Ton_{test_index if tests else i}.fig",
                                data=arr, compression="gzip")
                    except:
                        logging.info("Exception in converting file to hdf5")
                        continue

        # check that hdf5 file was written properly
        with hdf5.File(f"{result_folder}/{name}") as hdf5_file:
            assert all(map(len, hdf5_file.values()))


class Individual:

    def __init__(self,
                 pvalue=0.0,
                 pvalue_amplitude=0.0,
                 pvalue_times=0.0,
                 pvalue_dt=0.0,
                 peaks_number=0.0,
                 id=0,
                 origin="",
                 population_number=0,
                 gen=None):

        if gen is None:
            gen = []

        self.pvalue = pvalue
        self.pvalue_amplitude = pvalue_amplitude
        self.pvalue_times = pvalue_times
        self.peaks_number = peaks_number
        self.pvalue_dt = pvalue_dt
        self.id = id
        self.origin = origin
        self.population_number = population_number
        self.gen = gen
        self.weights = []
        self.delays = []

    def __str__(self):
        return f"Individual with p-value = {self.pvalue}, p-value amplitude = {self.pvalue_amplitude}, " \
            f"p-value times = {self.pvalue_times}, peaks number = {self.peaks_number}, origin from {self.origin}\n "

    def __repr__(self):
        return f"Individual with p-value = {self.pvalue}, p-value amplitude = {self.pvalue_amplitude}, " \
            f"p-value times = {self.pvalue_times}, peaks number = {self.peaks_number}, origin from {self.origin}\n"

    def __eq__(self, other):
        return self.pvalue_amplitude * self.pvalue * self.pvalue_dt == other.pvalue_amplitude * other.pvalue * other.pvalue_dt

    def __gt__(self, other):
        return self.pvalue_amplitude * self.pvalue * self.pvalue_dt > other.pvalue_amplitude * other.pvalue * other.pvalue_dt

    def __copy__(self):

        new_individual = Individual()

        for g in self.gen:
            new_individual.gen.append(g)

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        # new_individual.pvalue_dt = self.pvalue_dt
        # new_individual.pvalue = self.pvalue
        # new_individual.pvalue_times = self.pvalue_times
        # new_individual.pvalue_amplitude = self.pvalue_amplitude
        # new_individual.population_number = self.population_number
        # new_individual.peaks_number = self.peaks_number
        new_individual.origin = self.origin

        return new_individual

    def __len__(self):
        return len(self.gen)

    def normalize(self):
        for i in range(len(self)):

            if i < N:

                if self.gen[i] < low_weights[i]:
                    self.gen[i] = low_weights[i]
                if self.gen[i] > max_weights[i]:
                    self.gen[i] = max_weights[i]

            else:

                if self.gen[i] < low_delay:
                    self.gen[i] = low_delay
                if self.gen[i] > max_delay:
                    self.gen[i] = max_delay

    @staticmethod
    def encode(individual_to_decode):
        return {'Population number': individual_to_decode.population_number,
                'p-value 2d': individual_to_decode.pvalue,
                'p-value amplitude': individual_to_decode.pvalue_amplitude,
                'p-value times': individual_to_decode.pvalue_times,
                'p-value dt': individual_to_decode.pvalue_dt,
                'peaks number': individual_to_decode.peaks_number,
                'origin': individual_to_decode.origin,
                'weights and delays': individual_to_decode.gen}

    @staticmethod
    def decode(dct):
        return Individual(pvalue=dct['p-value 2d'], pvalue_amplitude=dct['p-value amplitude'],
                          peaks_number=dct['peaks number'], pvalue_dt=dct['p-value dt'],
                          pvalue_times=dct['p-value times'],
                          population_number=dct['Population number'], gen=dct['weights and delays'],
                          origin=dct['origin'])

    @staticmethod
    def format_weight(weight):
        return float("{0:.3f}".format(weight))

    @staticmethod
    def format_delay(delay):
        return float("{0:.1f}".format(delay))

    def is_correct(self):
        # ~ p-value_times != 0 and p-value_amplitude != 0 and pvalue != 0
        return self.pvalue_amplitude * self.pvalue_times * self.pvalue * self.pvalue_dt != 0 and self.peaks_number >= DataSettings.min_peaks_number

    def set_weight(self, min_weight, max_weight):
        self.weights.append(Individual().format_weight(random.uniform(min_weight, max_weight)))
        if len(max_weights) < N and len(low_weights) < N:
            max_weights.append(max_weight)
            low_weights.append(min_weight)

    def set_delay(self):
        self.delays.append(Individual().format_delay(random.uniform(low_delay, max_delay)))

    # init individual with random weights and delays
    def init(self):

        # Es ~ OMs
        for i in range(5):
            self.set_weight(0.01, 0.6)

        # CVs - OMs
        for i in range(16):
            self.set_weight(0.06, 2.2)

        # output to Flexor another OM
        for i in range(4):
            self.set_weight(0.001, 0.02)

        # output to eIP
        for i in range(10):
            self.set_weight(0.1, 6)

        for i in range(40):
            self.set_weight(0.05, 2)

        for i in range(15):
            self.set_weight(0.01, 0.3)

        for i in range(2):
            self.set_weight(0.005, 0.07)

        for i in range(4):
            self.set_weight(0.0005, 0.002)

        # eIP ~ MN
        for i in range(2):
            self.set_weight(1, 15)

        for i in range(6):
            self.set_weight(0, 0)

        # init delays
        for i in range(N):
            self.set_delay()

        self.gen = self.weights + self.delays
        self.origin = "first init "


class Data:
    path_to_files = f"{path}/files"
    path_to_dat_folder = f"{path}/dat"

    log_files = [f"{path_to_files}/history.dat", f"{path_to_files}/bests_pvalue.dat",
                 f"{path_to_files}/log.dat", f"{path_to_files}/log_of_bests.dat"]

    files = []

    for i in range(4):
        files.append(f"{path_to_dat_folder}/{i}/gras_E_PLT_21cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_F_PLT_21cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_E_PLT_13.5cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_F_PLT_13.5cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_E_PLT_6cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_F_PLT_6cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/a.txt")
        files.append(f"{path_to_dat_folder}/{i}/t.txt")
        files.append(f"{path_to_dat_folder}/{i}/d2.txt")
        files.append(f"{path_to_dat_folder}/{i}/peaks.txt")
        files.append(f"{path_to_dat_folder}/{i}/dtt.txt")
        files.append(f"{path}/pickle/{i}/gras_PLT_6cms_40Hz_2pedal_0.025step.pickle")
        files.append(f"{path}/pickle/{i}/gras_PLT_13.5cms_40Hz_2pedal_0.025step.pickle")
        files.append(f"{path}/pickle/{i}/gras_PLT_21cms_40Hz_2pedal_0.025step.pickle")
        for j in range(5):
            files.append(f"{path_to_dat_folder}/{i}/{j}_MN_E.dat")
            files.append(f"{path_to_dat_folder}/{i}/{j}_MN_F.dat")

    @staticmethod
    def delete(files_arr):
        for file in files_arr:
            if os.path.isfile(file):
                print(f"Deleted {file}")
                os.remove(f"{file}")

    @staticmethod
    def delete_files():
        Data().delete(Data.files)

    @staticmethod
    def delete_all_files():
        Data().delete(Data.log_files + Data.files)


class Population:

    def __init__(self):
        self.individuals = []

    def add_individual(self, individual):
        self.individuals.append(individual)

    def __len__(self):
        return len(self.individuals)

    def first_init(self, known=False):
        self.first_init_known() if known else self.first_init_unknown()

    # init N_individuals_in_first_init individuals for first population
    def first_init_unknown(self):
        for i in range(DataSettings.N_individuals_in_first_init):
            individual = Individual()
            individual.init()
            self.add_individual(individual)
            individual.id = i

    def first_init_known(self):
        individual = Individual()
        individual.init()

        with open(DataSettings.path_to_json_file_with_individuals) as file:
            all_lines = file.readlines()

        for line in all_lines:
            self.add_individual(json.loads(line, object_hook=Individual.decode))

    def normalize_hyperparameters(self):
        final_population = Population()

        for individual in self.individuals:
            individual.normalize()

            final_population.add_individual(individual)

        return final_population


class Fitness:

    # calculate fitness function for instance of Invididual class
    @staticmethod
    def calculate_fitness(individuals, num_population):

        # converting 4 results data to hdf5
        for i in range(4):
            convert_to_hdf5(f"{path}/dat/{i}", tests=DataSettings.is_tests)

        # get p-value for 4 individuals
        start = time.time()
        get_4_pvalue()
        print(f"4 p-value calculated {time.time() - start}")

        # set p-value to this individuals
        for i in range(len(individuals)):
            individual = individuals[i]

            ampls = open(f'{path}/dat/{i}/a.txt')
            times = open(f'{path}/dat/{i}/t.txt')
            d2 = open(f'{path}/dat/{i}/d2.txt')
            peaks = open(f'{path}/dat/{i}/peaks.txt')
            dt = open(f'{path}/dat/{i}/dtt.txt')

            try:
                individual.pvalue_amplitude = float(ampls.readline())
                individual.pvalue_times = float(times.readline())
                individual.pvalue = float(d2.readline())
                individual.peaks_number = float(peaks.readline())
                pval_dt = float(dt.readline())

                if pval_dt == pval_dt:
                    individual.pvalue_dt = pval_dt
                else:
                    individual.pvalue_dt = 0.0

                fnameE = f"{path}/dat/{i}/gras_E_PLT_{speed}cms_40Hz_2pedal_0.025step.hdf5"
                fnameF = f"{path}/dat/{i}/gras_F_PLT_{speed}cms_40Hz_2pedal_0.025step.hdf5"

                Fitness.write_pvalue(individual, num_population)

                if individual.pvalue_amplitude >= 0.05 and individual.pvalue_times >= 0.05 and individual.pvalue >= 0.05:
                    os.rename(f"{fnameE}",
                              f"{path}/dat/{i}/pampl{individual.pvalue_amplitude}pt{individual.pvalue_times}2d{individual.pvalue}_E_a_and_t_and_2d")
                    os.rename(f"{fnameF}",
                              f"{path}/dat/{i}/pampl{individual.pvalue_amplitude}pt{individual.pvalue_times}2d{individual.pvalue}_F_a_and_t_and_2d")
                elif individual.pvalue_amplitude >= 0.05 and individual.pvalue_times >= 0.05:
                    os.rename(f"{fnameE}",
                              f"{path}/dat/{i}/pampl{individual.pvalue_amplitude}pt{individual.pvalue_times}_E_a_and_t")
                    os.rename(f"{fnameF}",
                              f"{path}/dat/{i}/pampl{individual.pvalue_amplitude}pt{individual.pvalue_times}_F_a_and_t")
                elif individual.pvalue_amplitude >= 0.05:
                    os.rename(f"{fnameE}", f"{path}/dat/{i}/{individual.pvalue_amplitude}_E_ampl")
                    os.rename(f"{fnameF}", f"{individual.pvalue_amplitude}_F_ampl")
                elif individual.pvalue >= 0.05:
                    os.rename(f"{fnameE}", f"{path}/dat/{i}/{individual.pvalue}_E_2d")
                    os.rename(f"{fnameF}", f"{individual.pvalue}_F_2d")

            except:
                print("Error in calculate_fitness")
                continue

        Data.delete_files()

    @staticmethod
    def write_pvalue(individual, number):
        individual.population_number = number

        file = open(f'{path}/files/history.dat', 'a')
        file.write(f"{json.dumps(individual, default=Individual.encode)}\n")
        file.close()

    # choose best value of fitness function for population
    @staticmethod
    def best_fitness(current_population):
        return max(current_population.individuals)


class Breeding:

    @staticmethod
    def crossover(individual_1, individual_2):

        crossover_point = random.randint(0, len(individual_1))

        new_individual_1 = Individual()
        new_individual_1.gen = individual_1.gen[:crossover_point] + individual_2.gen[crossover_point:]
        new_individual_1.weights = new_individual_1.gen[:int(len(new_individual_1) / 2)]
        new_individual_1.delays = new_individual_1.gen[int(len(new_individual_1) / 2):]

        new_individual_2 = Individual()
        new_individual_2.gen = individual_2.gen[:crossover_point] + individual_1.gen[crossover_point:]
        new_individual_2.weights = new_individual_1.gen[:int(len(new_individual_1) / 2)]
        new_individual_2.delays = new_individual_1.gen[int(len(new_individual_1) / 2):]

        new_individual_1.origin, new_individual_2.origin = "crossover1", "crossover2"

        return new_individual_1, new_individual_2

    @staticmethod
    def mutation2(individual):

        new_individual = individual.__copy__()

        n = random.randint(1, 2)

        for index in range(len(individual)):
            if index % 2 == n - 1:
                low, high = Breeding.get_low_high(individual.gen[index])
                if index < N:
                    new_individual.gen[index] = Individual().format_weight(random.uniform(low, high))
                else:
                    new_individual.gen[index] = Individual().format_delay(random.uniform(low, high))

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        new_individual.origin += " mutation2"

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

    @staticmethod
    def mutation3(individual):

        new_individual = individual.__copy__()

        for index in range(len(individual)):
            n = random.randint(0, 100)
            if n < 50:
                low, high = Breeding.get_low_high(individual.gen[index])
                if index < N:
                    new_individual.gen[index] = Individual().format_weight(random.uniform(low, high))
                else:
                    new_individual.gen[index] = Individual().format_delay(random.uniform(low, high))

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        new_individual.origin += " mutation3"

        return new_individual

    @staticmethod
    def mutation4(individual):

        new_individual = individual.__copy__()

        for index in range(len(individual)):
            n = random.randint(0, 100)
            if n < 50:
                m = random.randint(2, 10)
                g = individual.gen[index]
                low, high = g - g / m, g + g / m
                if index < N:
                    new_individual.gen[index] = Individual().format_weight(random.uniform(low, high))
                else:
                    new_individual.gen[index] = Individual().format_delay(random.uniform(low, high))

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        new_individual.origin += " mutation4"

        return new_individual

    @staticmethod
    def mutation(individual):

        new_individual = individual.__copy__()
        mutation_point = random.randint(0, len(individual))

        for index in range(mutation_point):
            mean = new_individual.gen[index]
            low, high = Breeding.get_low_high(mean)
            if index < N:
                new_individual.gen[index] = Individual().format_weight(random.uniform(low, high))
            else:
                new_individual.gen[index] = Individual().format_delay(random.uniform(low, high))

        new_individual.origin += " mutation"

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

                s = f"Skip individual because {str(current_population.individuals[index])}\n"

                print(s)
                logg_string += s

        logg_string += f"Skiped {counter} individuals\n"

        file = open(f"{path}/files/log.dat", 'a')
        file.write(logg_string)
        file.close()

        # sort this individuals
        arr = sorted(newPopulation.individuals, reverse=True)

        return arr[:N_how_much_we_choose] if len(arr) > N_how_much_we_choose else arr

    @staticmethod
    def calculate(current_population, is_calculate_again=False):

        arr1 = []

        if is_calculate_again:
            b = int(len(current_population) / 4)
            cp = current_population[:b * 4]

            arr1.append(current_population[b * 4:])

        else:
            individuals_in_current_population_with_uknown_pvalue = []
            individuals_in_current_population_with_uknown_pvalue_count = 0
            len_current_population = 0

            for individual in current_population:
                len_current_population += 1
                if individual.pvalue_times == 0 and individual.pvalue_amplitude == 0:
                    individuals_in_current_population_with_uknown_pvalue.append(individual)
                    individuals_in_current_population_with_uknown_pvalue_count += 1

            b = int(len(individuals_in_current_population_with_uknown_pvalue) / 4)
            cp = individuals_in_current_population_with_uknown_pvalue[:b * 4]

            arr1.append(individuals_in_current_population_with_uknown_pvalue[b * 4:])

        k = 0
        while True:
            arr = []
            if k >= len(cp):
                break
            for i in range(4):
                if k >= len(cp):
                    break
                arr.append(cp[k])
                k += 1
            arr1.append(arr)

        return arr1

    @staticmethod
    def calculate_tests_result(current_population, number):

        arr1 = Breeding.calculate(current_population, is_calculate_again=False)

        for i in range(len(arr1)):
            Build.run_tests(arr1[i])
            time.sleep(0.2)
            Fitness.calculate_fitness(arr1[i], number)


class Evolution:
    terminate_algorithm = False

    @staticmethod
    def start_gen_algorithm(current_population, number):

        Breeding.calculate_tests_result(current_population.individuals, number)
        bests_individuals_in_current_population = Breeding.select(current_population)

        Debug.log_of_bests(bests_individuals_in_current_population, number)

        new_population = Population()

        length_population = len(bests_individuals_in_current_population)

        # each with each
        for i in range(length_population):
            individual_1 = bests_individuals_in_current_population[i]
            j = i
            while j + 1 < length_population:
                j += 1
                individual_2 = bests_individuals_in_current_population[j]
                crossover_individual_1, crossover_individual_2 = Breeding.crossover(individual_1, individual_2)
                new_population.add_individual(crossover_individual_1)
                new_population.add_individual(crossover_individual_2)

        for individual in new_population.individuals:
            for i in range(len(individual.gen)):
                individual.gen[i] = float(individual.gen[i])

        for individual in new_population.individuals:
            r = random.randint(0, 100)
            if r >= 50:
                mn = random.randint(0, 3)
                if mn == 0:
                    new_population.add_individual(Breeding.mutation(individual))
                elif mn == 1:
                    new_population.add_individual(Breeding.mutation2(individual))
                elif mn == 2:
                    new_population.add_individual(Breeding.mutation3(individual))
                elif mn == 3:
                    new_population.add_individual(Breeding.mutation4(individual))

        for individual in bests_individuals_in_current_population:
            r = random.randint(0, 100)
            if r >= 50:
                new_population.add_individual(Breeding.mutation3(individual))

        for individual in bests_individuals_in_current_population:
            r = random.randint(0, 100)
            if r >= 50:
                new_population.add_individual(Breeding.mutation2(individual))

        for individual in bests_individuals_in_current_population:
            r = random.randint(0, 100)
            if r >= 50:
                new_population.add_individual(Breeding.mutation4(individual))

        for individual in bests_individuals_in_current_population:
            r = random.randint(0, 100)
            if r >= 50:
                new_population.add_individual(Breeding.mutation(individual))

        # add old best individuals if new will be worst
        for individual in bests_individuals_in_current_population:
            if individual.is_correct():
                new_population.add_individual(individual)

        final_population = new_population.normalize_hyperparameters()

        del current_population
        del new_population

        return final_population


class Debug:

    @staticmethod
    def log_of_bests(individuals, number):
        best_individual = individuals[0]
        best_individual.population_number = number

        file_with_log_of_bests = open(f'{path}/files/log_of_bests.dat', 'a')
        file_with_log_of_bests.write(f"{json.dumps(individuals, default=Individual.encode)}\n")
        file_with_log_of_bests.close()

        file = open(f'{path}/files/bests_pvalue.dat', 'a')
        file.write(f"{json.dumps(best_individual, default=Individual.encode)}\n")
        file.close()


if __name__ == "__main__":

    Data().delete_all_files()

    Build.compile()

    # first initialization to population
    population = Population()
    population.first_init(known=True)
    population_number = 1

    # TODO add deleting log.err, log.out each 10 population

    while not Evolution.terminate_algorithm:
        print(f"Population number = {population_number}")

        next_population = Evolution.start_gen_algorithm(population, population_number)

        population = next_population

        population_number += 1
