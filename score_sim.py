import os 
import sys
sys.path.append("/share/hongliang/")
import phylopandas.phylopandas as ph
import pandas as pd

tar_dir = sys.argv[1]+'/'
#tar_dir = "/share/hongliang/results/result-E0.1-incE0.001/"
#tar_dir = "/share/hongliang/out-test/"
base_dir = "/user/hongliang/mydpr/c1000_msa/"

a3m_list = os.listdir(tar_dir)
a3m_list.sort()

result1 = 0.0
result2 = 0.0

for fname in a3m_list:
    tar_df = ph.read_fasta(tar_dir+fname, use_uids=False)
    tar_id = tar_df['id'].map(lambda x: x.split('/')[0]).drop_duplicates()
    base_df = ph.read_fasta(base_dir+fname, use_uids=False)
    base_id = base_df['id'].map(lambda x: x.split('/')[0]).drop_duplicates()
    tar_num = tar_id.shape[0]
    base_num = base_id.shape[0]
    tot_id = pd.concat([tar_id, base_id]).drop_duplicates()
    cov_num = tar_num + base_num - tot_id.shape[0]
    tar_num = tar_num if tar_num>0 else 1
    base_num = base_num if base_num>0 else 1
    result1 += cov_num/tar_num
    result2 += cov_num/base_num

result1 /= len(a3m_list)
result2 /= len(a3m_list)

print(tar_dir)
print("%.3f = %.3f + %.3f"%(result1+result2, result1, result2))
