from analysis.PCA import read_data

all_pack = []
colors = iter(["#275b78", "#287a72", "#f2aa2e", "#472650"])

bio_folder_foot = "/home/anna/PycharmProjects/LAB/memristive-spinal-cord/bio-data/hdf5/foot"
bio_folder_toe = "/home/anna/PycharmProjects/LAB/memristive-spinal-cord/bio-data/hdf5/toe"

bio_pack_foot = ["bio_E_6cms_40Hz_i100_2pedal_no5ht_T", "bio_E_21cms_40Hz_i100_2pedal_no5ht_T",
            "bio_F_6cms_40Hz_i100_2pedal_no5ht_T", "bio_F_21cms_40Hz_i100_2pedal_no5ht_T"]
bio_pack_toe = ["bio_E_21cms_40Hz_i100_2pedal_no5ht_T", "bio_F_21cms_40Hz_i100_2pedal_no5ht_T"]

gras_folder_foot = "/home/anna/PycharmProjects/LAB/memristive-spinal-cord/GRAS/hdf5/foot"
gras_folder_toe = "/home/anna/PycharmProjects/LAB/memristive-spinal-cord/GRAS/hdf5/toe"

gras_pack_foot = ["gras_E_6cms_40Hz_i100_2pedal_no5ht_T", "gras_E_21cms_40Hz_i100_2pedal_no5ht_T",
                  "gras_F_6cms_40Hz_i100_2pedal_no5ht_T", "gras_F_21cms_40Hz_i100_2pedal_no5ht_T"]

gras_pack_toe = ["gras_E_21cms_40Hz_i100_2pedal_no5ht_T","gras_F_21cms_40Hz_i100_2pedal_no5ht_T"]

pack = bio_pack_foot
folder = bio_folder_foot

for data_name in pack:
	path_extensor = f"{folder}/{data_name}.hdf5"
	path_flexor = f"{folder}/{data_name.replace('_E_', '_F_')}.hdf5"
	# check if it is a bio data -- use another function
	if "bio_" in data_name:
		e_dataset = read_data(path_extensor)
		f_dataset = read_data(path_flexor)
	# simulation data computes by the common function
	else:
		# calculate extensor borders
		extensor_begin = 0
		extensor_end = 6000 if "21cms" in data_name else 30000
		# calculate flexor borders
		flexor_begin = extensor_end
		flexor_end = extensor_end + 5000
		# use native funcion for get necessary data
		e_dataset = select_slices(path_extensor, extensor_begin, extensor_end)
		f_dataset = select_slices(path_flexor, flexor_begin, flexor_end)
	# prepare each data (stepping, centering, normalization)
	e_prepared_data = prepare_data(e_dataset)
	f_prepared_data = prepare_data(f_dataset)
	# get latencies, amplitudes and begining of poly answers
	lat_per_slice, amp_per_slice, mono_per_slice = get_lat_amp(e_prepared_data,
	                                                           ees_hz=40, data_step=0.25, debugging=False)
	# get number of peaks per slice
	peaks_per_slice = get_peaks(e_prepared_data, herz=40, step=0.25)[7]
