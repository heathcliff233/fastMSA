import subprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

msa_dir = sys.argv[1]
out_name = sys.argv[2]

msa_list = os.listdir(msa_dir)
line_list = np.array([int(subprocess.getoutput("wc -l %s" % (msa_dir+file_name)).split()[0])/2 for file_name in msa_list])
bins = [i for i in range(15)]
plt.hist(np.log(line_list), bins=bins)
plt.xlim(0, 15)
plt.ylim(0, 45)
plt.xlabel("ln {num of sequences in MSA}")
plt.ylabel("num of MSAs")
#plt.title("Histogram of CASP14 filtered ground truth")
plt.title("Histogram of CASP14 MSA in top 1M prediction")
plt.savefig("/user/hongliang/mydpr/counter/%s.png"%out_name)



