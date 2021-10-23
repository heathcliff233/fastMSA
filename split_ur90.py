import sys
import os
sys.path.append('/share/hongliang/')
import phylopandas.phylopandas as ph

fasta_path = "/ssdcache/zhengliangzhen/sequence_databases/uniref90_2020_03.fasta"
num_splits = 10
save_base = "/share/hongliang/ur90_split/split-"

df = ph.read_fasta(fasta_path, use_uids=False)
df_len = df.shape[0]
print("Finished reading UR90")
df = df.sample(frac=1.0)
df = df.reset_index(drop=True)
print("Shuffled data")

for i in range(num_splits):
    tmp_df = df.iloc[i*df_len//num_splits:(i+1)*df_len//num_splits]
    tmp_df.phylo.to_fasta_dev(save_base+str(i)+".fasta")
    print("\r Finish saving split %d / %d"%(i+1, num_splits))
