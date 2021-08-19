import numpy as np
import subprocess
from typing import Sequence, Tuple, List, Union

#path = "./testset/"
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"

def wc_count(file_name):
    #return 6
    out = subprocess.getoutput("wc -l %s" % file_name)
    res = int(out.split()[0])
    return res

def get_filename(sel_path: str) -> List[str]:
    nfile = np.genfromtxt(sel_path, dtype='str').T
    path_list = nfile[0]
    names = [str(name)+'.a3m' for name in path_list]
    lines = nfile[1].astype(np.int32).tolist()
    #if count_lines:
    #    lines = [wc_count(name) for name in names]
    #else:
    #    lines = []
    return names, lines
