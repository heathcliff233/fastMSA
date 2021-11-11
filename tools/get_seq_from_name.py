import numpy as np
from Bio import SeqIO

database_path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_seq/"
path_to_name = './v3-trial.txt'
path_to_file = path_to_name[:-4] + '-seq.fasta'
df = np.loadtxt(path_to_name, delimiter=' ', dtype='str')
num_name = df.shape[0]
df = df.T[0]
records = []
for i in df :
    file_path = database_path + i + '.seq'
    records.append(SeqIO.parse(file_path, 'fasta'))

with open(path_to_file, 'w') as f:
    for rec in records:
        SeqIO.write(rec, f, 'fasta')
