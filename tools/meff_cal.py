import subprocess
import os 
import sys

base_dir = '/share/hongliang/cameo-100-fseq/'
file_list = sorted(os.listdir(base_dir))

for fp in file_list:
    out1 = subprocess.getoutput("grep -o '>' %s | wc -l" % (base_dir+fp))
    out2 = subprocess.getoutput("/share/hongliang/meff -i %s" % (base_dir+fp))
    name = fp.split()[0]
    cnt = out1.split()[0]
    sim = out2.split()[0]
    print("%s, %s, %s" % (name, cnt, sim))

