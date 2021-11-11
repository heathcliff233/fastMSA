import numpy as np
import subprocess

path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
def wc_count(file_name):
    out = subprocess.getoutput("wc -l %s" % file_name)
    res = int(out.split()[0])
    return res
    
def get_filename(sel_path: str, out_path: str):
    path_list = np.genfromtxt(sel_path, dtype='str')
    for rec in range(path_list.shape[0]):
        path_list[rec][1] = wc_count(path+str(path_list[rec][0])+".a3m")
    np.savetxt(out_path, path_list, delimiter=' ', fmt="%s")

base = './split/'
choice = ['split_dataset_test', 'split_dataset_train', 'split_dataset_valid']
for i in choice:
    get_filename(base+i+'.txt', base+i[-5:]+'.txt')
    print("finish ", i)

