from pkg_resources import resource_filename
filepath = resource_filename('spinal_cord', 'autoparams.csv')

with open(filepath, 'w') as params_file:
    index = 0
    for aff_ia_moto in range(1, 5, 1):
        for aff_ia_ia in range(1, 5, 1):
            for aff_ii_ii in range(1, 10, 1):
                for ii_moto in range(1, 5, 1):
                    for ia_ia in range(1, -5, -1):
                        for ia_moto in range(0, -5, -1):
                            index += 1
                            print(index)
                            params_file.write('{}_{}_{}_{}_{}_{}_{}\n'.format(
                                index,
                                aff_ia_moto,
                                aff_ia_ia,
                                aff_ii_ii,
                                ii_moto,
                                ia_ia,
                                ia_moto))
