from Bio import SeqIO
import os
tar_dir = './c1000_msa/'
src_dir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"

file_list = os.listdir(src_dir)
total_files = len(file_list)
cnt = 0

for f in file_list:
    src_seqs = list(SeqIO.parse(src_dir+f, "fasta"))
    tar_seqs = src_seqs[:1000]
    with open(tar_dir+f, "w") as output_handle:
        SeqIO.write(tar_seqs, output_handle, "fasta")
    cnt += 1
    print("\r Finished %-5d/%d"%(cnt, total_files), end="")
