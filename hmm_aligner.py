import os
import sys
file_list = os.listdir('pred-202108241047-20000')
file_list.sort()
qjackhmmer = "/share/wangsheng/GitBucket/alphafold2_sheng/alphafold2/util/qjackhmmer"
cnt = 0
for file in file_list:
    pref = file[:-4]
    args = " -B hmmalign20000-v2/%s.a3m --F1 0.0005 --F2 5e-05 --F3 5e-07 --incE 0.0001 -E 0.0001 --cpu 8 -N 1 /share/wangsheng/train_test_data/cath35_20201021/cath35_seq/%s.seq pred-202108241047-20000/%s.a3m | grep -E \'New targets included:|Target sequences:\'"%(pref, pref, pref)
    cmd = qjackhmmer + args
    print(pref)
    os.system(cmd)
    cnt += 1
    if cnt%10==0:
        print("\r Processed ", cnt//10,'%', file=sys.stderr, end="")
print("", file=sys.stderr)
