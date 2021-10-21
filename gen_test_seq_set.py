import os
import sys

seq_dir = './cameo_test_small/'
save_path = './cameo_test_small.fasta'

file_list = os.listdir(seq_dir)
file_list.sort()

with open(save_path, 'w') as f:
    for fp in file_list:
        first_line = True
        with open(seq_dir + fp, 'r') as hd:
            for line in hd.readlines():
                if first_line:
                    f.write('>'+fp[:-6]+'\n')
                    first_line = False
                else:
                    f.write(line+'\n')

