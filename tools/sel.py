import os
import subprocess
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
save = "./testset/"
files = os.listdir(path)
cnt = 0
for file in files :
    if cnt >= 1000:
        break
    if cnt%10==0:
        print("have finished %d / 1000", cnt)
    if ".a3m" in file:
        out = subprocess.getoutput("wc -l %s" % path+file)
        res = int(out.split()[0])
        if res > 2048 :
            cmd = "head -2048 %s > %s" %(path+file, save+file)
            #subprocess.run(["head", "-2048", path+file, ">", save+file])
            os.system(cmd)
            cnt += 1
