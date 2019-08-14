from GRAS.tests_runner import convert_to_hdf5
from GRAS.tests_runner import run_tests
from GRAS.tests_runner import plot_results
import os, shutil, glob

tests_number, cms, ees, inh, ped, ht5, save_all = range(7)

main_src_folder = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat"

folder_for_saving_results = ""

cms_arr = [6, 15, 21]
ees_arr = [i * 5 + 5 for i in range(0, 19)]
inh_arr = [0, 50, 100]
ped_arr = [2, 4]

def run_all_tests(tests_number_for_test=10, cms_for_test=[21], ees_for_test=[40],
                  inh_for_test=[100], ped_for_test=[2],
                  ht5_for_test=0, save_all_for_test=0, toe=False, air=False, STDP=False):

    src_fldr = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat"
    script_place = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/"

    if STDP:
        script_place = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/with_STDP"
        if not os.path.isdir(f"{src_fldr}/STDP"):
            os.mkdir(f"{src_fldr}/STDP")
        src_fldr = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat/STDP"

    if toe:
        script_place = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/toe"
        if not os.path.isdir(f"{src_fldr}/toe"):
            os.mkdir(f"{src_fldr}/toe")
        src_fldr = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat/toe"

    if air:
        script_place = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat/air"
        if not os.path.isdir(f"{src_fldr}/air"):
            os.mkdir(f"{src_fldr}/air")
        src_fldr = "/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat/air"

    for c_m_s in cms_for_test:
        for e_e_s in ees_for_test:
            for i_n_h in inh_for_test:
                for p_e_d in ped_for_test:

                    args = {tests_number: tests_number_for_test,
                            cms: c_m_s,
                            ees: e_e_s,
                            inh: i_n_h,
                            ped: p_e_d,
                            ht5: ht5_for_test,
                            save_all: save_all_for_test}

                    # creating folder for saving results of tests
                    if len(ees_for_test) == 1 and e_e_s == 40:
                        folder_for_saving_results = f"{src_fldr}/cms_{c_m_s},ees_{e_e_s},inh_{i_n_h},ped_{p_e_d}"
                    else:
                        folder_for_saving_results = f"{src_fldr}/differents_hz"

                    if not os.path.isdir(folder_for_saving_results):
                        os.mkdir(folder_for_saving_results)

                    run_tests(script_place, args)

                    files = glob.iglob(os.path.join(main_src_folder, "*.dat"))
                    for file in files:
                        if os.path.isfile(file):
                            shutil.copy2(file, folder_for_saving_results)
                            os.remove(f"{file}")

                    convert_to_hdf5(folder_for_saving_results, args)

                    if e_e_s == 40 and len(ees_for_test) == 1:
                        plot_results(folder_for_saving_results, ees_hz=args[ees])

                    files = glob.iglob(os.path.join(folder_for_saving_results, "*.dat"))
                    for file in files:
                        if os.path.isfile(file):
                            os.remove(f"{file}")


if __name__ == "__main__":

    run_all_tests(tests_number_for_test=25,
                  cms_for_test=[21],
                  ees_for_test=[40],
                  inh_for_test=[100],
                  ped_for_test=[2],
                  ht5_for_test=0,
                  save_all_for_test=0,
                  toe=False,
                  air=False,
                  STDP=False)