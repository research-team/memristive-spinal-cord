import json
import unittest
import h5py
import random
import numpy as np

from python_scripts.genetic_algotithm import Individual
from python_scripts.genetic_algotithm import Population
from python_scripts.genetic_algotithm import Debug
from python_scripts.genetic_algotithm import Fitness
from python_scripts.genetic_algotithm import convert_to_hdf5
from python_scripts.genetic_algotithm import max_weights, low_weights
from python_scripts.genetic_algotithm import Breeding

# python -m unittest test.py


class TestJson(unittest.TestCase):

    i1 = Individual(peaks_number=100,
                    pvalue_amplitude=0.1,
                    pvalue=0.001,
                    pvalue_times=1,
                    pvalue_dt=0.01,
                    population_number=10,
                    gen=[1, 2, 3],
                    origin="mut")

    i2 = Individual(peaks_number=100,
                    pvalue_amplitude=0.2,
                    pvalue=0.001,
                    pvalue_times=1,
                    pvalue_dt=0.01,
                    population_number=10,
                    gen=[1, 2, 3],
                    origin="mut")

    i3 = Individual(peaks_number=100,
                    pvalue_amplitude=0.3,
                    pvalue=0.001,
                    pvalue_times=1,
                    pvalue_dt=0.01,
                    population_number=10,
                    gen=[1, 2, 3],
                    origin="mut")

    def testEncodeDecode(self):

        json_string = f"{json.dumps(TestJson.i1, default=Individual.encode)}\n"

        i2 = json.loads(json_string, object_hook=Individual.decode)

        self.assertEqual(TestJson.i1.peaks_number, i2.peaks_number)
        self.assertEqual(TestJson.i1.pvalue_amplitude, i2.pvalue_amplitude)
        self.assertEqual(TestJson.i1.pvalue, i2.pvalue)
        self.assertEqual(TestJson.i1.pvalue_times, i2.pvalue_times)
        self.assertEqual(TestJson.i1.pvalue_dt, i2.pvalue_dt)
        self.assertEqual(TestJson.i1.population_number, i2.population_number)
        self.assertEqual(TestJson.i1.gen, i2.gen)
        self.assertEqual(TestJson.i1.origin, i2.origin)

    def test_array(self):
        arr = [TestJson.i1, TestJson.i1, TestJson.i1]
        json_string = json.dumps(arr, default=Individual.encode)
        back_arr = json.loads(json_string, object_hook=Individual.decode)
        self.assertEqual(arr, back_arr)


class TestFF(unittest.TestCase):

    def test_is_correct(self):
        i1 = Individual(pvalue_times=0.1, pvalue=0.01, pvalue_amplitude=0.5, pvalue_dt=0.6, peaks_number=100)
        i2 = Individual(pvalue_times=0.1, pvalue=0.01, pvalue_amplitude=0.5, pvalue_dt=0.6, peaks_number=30)
        i3 = Individual(pvalue_times=0.1, pvalue=0.01, pvalue_amplitude=0.0, pvalue_dt=0.6, peaks_number=100)
        i4 = Individual(pvalue_times=0.1, pvalue=0.0, pvalue_amplitude=0.5, pvalue_dt=0.6, peaks_number=100)
        i5 = Individual(pvalue_times=0.1, pvalue=0.01, pvalue_amplitude=0.5, pvalue_dt=0.0, peaks_number=100)
        i6 = Individual(pvalue_times=0.0, pvalue=0.01, pvalue_amplitude=0.5, pvalue_dt=0.6, peaks_number=100)
        self.assertTrue(i1.is_correct())
        self.assertFalse(i2.is_correct())
        self.assertFalse(i3.is_correct())
        self.assertFalse(i4.is_correct())
        self.assertFalse(i5.is_correct())
        self.assertFalse(i6.is_correct())


@unittest.skip
class TestDebug(unittest.TestCase):

    def test_save(self):
        curr_pop = [TestJson.i3, TestJson.i1, TestJson.i2]
        p = Population()
        p.individuals = curr_pop
        Debug.log_of_bests(curr_pop, 6)

        Fitness.write_pvalue(TestJson.i3, 3)


@unittest.skip
class TestConvert(unittest.TestCase):

    def test_convert(self):
        convert_to_hdf5("/home/yuliya/Desktop/tt")
        with h5py.File('/home/yuliya/Desktop/tt/gras_E_PLT_21cms_40Hz_2pedal_0.025step.hdf5', 'r') as f:
            print(f.keys())


class TestCopy(unittest.TestCase):

    def test(self):
        gen = TestJson.i1.gen
        for i in range(100):
            i = TestJson.i1.__copy__()
            self.assertEqual(i.gen, gen)
            self.assertNotEqual(i, TestJson.i1)


@unittest.skip
class TestSort(unittest.TestCase):

    @staticmethod
    def f(x):
        return x.population_number == 2 and x.pvalue_dt == x.pvalue_dt

    def test(self):

        with open("../logs/history.dat") as file:
            a = file.readlines()

        arr = []

        for i in a:
            arr.append(json.loads(i, object_hook=Individual.decode))

        arr_pop2 = list(filter(TestSort.f, arr))

        arr = sorted(arr_pop2, reverse=True)
        print(arr)


class TestIndividual(unittest.TestCase):
    i1 = Individual()

    def test_normalize(self):
        self.i1.init()
        self.i1.gen[0] = 100
        self.i1.gen[1] = 0
        self.i1.normalize()
        arr = []
        for i in range(len(self.i1)):
            arr.append(self.i1.gen[i])

        arr[0] = max_weights[0]
        arr[1] = low_weights[1]
        self.assertEqual(self.i1.gen, arr)


class TestMutation(unittest.TestCase):

    def test_mutation(self):
        i = Individual()
        i.init()
        im = Breeding.mutation(i)

        self.assertNotEqual(i.gen, im.gen)

    def test_mutation2(self):
        i = Individual()
        i.init()
        im = Breeding.mutation2(i)

        self.assertNotEqual(i.gen, im.gen)

    def test_mutation3(self):
        i = Individual()
        i.init()
        im = Breeding.mutation3(i)

        self.assertNotEqual(i.gen, im.gen)

    def test_mutation4(self):
        i = Individual()
        i.init()
        im = Breeding.mutation4(i)

        self.assertNotEqual(i.gen, im.gen)

    def test_low_high(self):
        num = 100
        low_high1 = Breeding.get_low_high(num)
        low_high2 = Breeding.get_low_high(num)
        self.assertEqual(low_high1, low_high2)

    def test_t(self):
        cp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        arr1 = []
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

        print(arr1)

    def test_init(self):
        p = Population()
        p.first_init(known=True)
        g1 = p.individuals[0].gen
        p.normalize_hyperparameters()
        g2 = p.individuals[0].gen
        self.assertEqual(g1, g2)

    def test_low_w(self):
        print(low_weights)
        i = Individual()
        i.init()
        i2 = Individual()
        i2.init()
        print(len(low_weights))


if __name__ == "__main__":
    unittest.main()
