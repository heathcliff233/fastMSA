import os
import sys
import subprocess
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
files = os.listdir(path)
cnt = 0
tot = 0
for file in files:
    if ".a3m" in file:
        out = subprocess.getoutput("wc -l %s" % (path+file))
        res = int(out.split()[0])
        if res < 20 :
            cnt += 1
            print(path+file)
        tot += 1
print(cnt/tot, file=sys.stderr)
