from Bio.Align.Applications import ClustalOmegaCommandline
import subprocess
import os
from threading import Thread
import queue as Queue

inp = './pred-202108192300/'
otp = './align_pred/'
clustal = '/user/hongliang/clustalo-1.2.4-Ubuntu-x86_64'
files = os.listdir(inp)
files.sort()

def align_one(file_name):
    clustalomega_cline = ClustalOmegaCommandline(infile=inp+file_name, outfile=otp+file_name, verbose=True, auto=True)
    cmd = str(clustalomega_cline).split(" ")
    cmd[0] = clustal
    subprocess.run(cmd)

taskQueue = Queue.Queue()

def wrapper():
    while True:
        try:
            file_name = taskQueue.get(True, 5.5)
            if file_name is None:
                taskQueue.put(file_name)
                break
            align_one(file_name)
        except Queue.Empty:
            continue

threadsPool = [Thread(target=wrapper) for i in range(40)]

for thread in threadsPool:
    thread.start()

for file_name in files:
    taskQueue.put(file_name)

taskQueue.put(None)

for thread in threadsPool:
    thread.join()

print("finish")
