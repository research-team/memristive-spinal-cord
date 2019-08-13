import numpy as np
import h5py as hdf5


bio_path = '/home/yuliya/Desktop/STDP/bio-data/hdf5/bio_control_E_21cms_40Hz_i100_2pedal_no5ht_T_2017-09-05.hdf5'
gras_path = '/home/yuliya/Desktop/STDP/GRAS/matrix_solution/dat/cms_and_inh/cms: 21/cms: 21, inh: 100/gras_E_21cms_40Hz_i100_2pedal_no5ht_T.hdf5'


def corr_coef_2D(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def read_data(path, shrink=False):
    with hdf5.File(path) as file:
        if shrink:
            return np.array([d[:][::10][:1200] for d in file.values()])
        else:
            return np.array([d[:] for d in file.values()])


def split_to(data, mono=False, poly=False):
    mono_end_step = int(10 / 0.25)

    if not mono and not poly:
        raise Exception

    if poly:
        return data[:, mono_end_step:]

    if mono:
        return data[:, :mono_end_step]


bio_data = read_data(bio_path)
gras_data = read_data(gras_path, shrink=True)


gras_mono = split_to(gras_data, mono=True)
gras_poly = split_to(gras_data, poly=True)

bio_mono = split_to(bio_data, mono=True)
bio_poly = split_to(bio_data, poly=True)

mono_corr = np.array(corr_coef_2D(gras_mono, bio_mono))
poly_corr = np.array(corr_coef_2D(gras_poly, bio_poly))


import pylab as plt

for d in mono_corr, poly_corr:
    plt.boxplot(d.flatten())
    plt.plot([1] * (25 * 5), d.flatten(), '.')
    plt.show()

print(f"mono {mono_corr}")
print(f"poly {poly_corr}")