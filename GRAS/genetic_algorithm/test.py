import json
import unittest
import h5py

from python_scripts.genetic_algotithm import Individual
from python_scripts.genetic_algotithm import Population
from python_scripts.genetic_algotithm import Debug
from python_scripts.genetic_algotithm import Fitness
from python_scripts.genetic_algotithm import convert_to_hdf5


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
        for i in range(2):
            i = TestJson.i1.__copy__()
            print(i.gen)


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


if __name__ == "__main__":
    unittest.main()
