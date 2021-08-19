import os
path = "./testset/"
files = os.listdir(path)
files.sort()
for file in files :
    with open(path+file) as f:
        print(f.readline(), end="")
        print(f.readline(), end="")


