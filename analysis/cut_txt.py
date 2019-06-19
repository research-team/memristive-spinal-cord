lines_per_file = 37501440
smallfile = None
with open('../bio-data/notiception/5ht/4 Ser 2mkM (3).txt') as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = '5ht_half_4Ser2mkM_{}.txt'.format(lineno + lines_per_file)
            smallfile = open(small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()