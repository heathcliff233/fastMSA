import pickle
import os
import sys
import numpy as np
from typing import Sequence, Tuple, List, Union
import esm
import torch
from torch.utils.data import Dataset, DataLoader
from model import MyEncoder
from data import QueryDataset, SingleConverter, DistributedProxySampler
from train import do_embedding
from myutils import get_filename
import faiss
import linecache

BATCHSZ=8
#path = "./test.txt"
path = "./split/split_dataset_test.txt"
qdir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_seq/"
msadir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
ctx_dir = "./split_ebd/"
save_path = "./pred/"

def qencode(model, loader, device="cuda:0"):
    model.eval()
    res = []
    for i, (ids, toks) in enumerate(loader):
        toks = toks.to(device)
        with torch.no_grad():
            out = model.forward_once(toks)
        out = out.cpu().numpy()
        '''
        res.extend(
            [(ids[i], out[i]) for i in range(out.shape[0])]
        )
        '''
        res.extend(
            [out[i] for i in range(out.shape[0])]
        )
    return np.stack(res, axis=0)


def get_idx(idx, lst):
    cnt = 0
    tot = len(lst)
    while idx >= lst[cnt] and idx <= (tot-1):
        cnt += 1
    cnt -= 1
    return cnt, idx-lst[cnt]


if __name__ == "__main__":

    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    model = MyEncoder(encoder, 0)
    prev = torch.load('./split_train/39.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    device = torch.device("cuda:0")
    model = model.to(device)
    print("loaded model", file=sys.stderr)

    batch_converter = SingleConverter(alphabet)
    path_list = np.genfromtxt(path, dtype='str').T[0]
    path_list.sort()
    query_list = [qdir+str(name)+'.seq' for name in path_list]
    dataset = QueryDataset(query_list)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=batch_converter)
    encoded = qencode(model, dataloader, device=device)
    print(encoded.shape, file=sys.stderr)
    print("encoded query", file=sys.stderr)

    ctx_list = os.listdir('./split_ebd')
    ctx_list.sort()
    ttpath = './split/test.txt'
    file_list, lines = get_filename(ttpath)
    
    sorted_id = sorted(range(len(file_list)), key=lambda k: file_list[k])
    file_list.sort()
    src_list = [msadir+file for file in file_list]
    lines = [lines[i] for i in sorted_id]
    pctg = [0]*len(lines)
    for i in range(1, len(lines)):
        pctg[i] = pctg[i-1] + lines[i-1]//2
    buffer = []

    index = faiss.IndexFlatIP(768)
    cnt = 0
    tot_ctx = len(ctx_list)
    for i in ctx_list:
        with open('./split_ebd/'+i, "rb") as reader:
            whole_doc = pickle.load(reader)
            for name, tok in whole_doc:
                buffer.append(tok)
            index.add(np.stack(buffer, axis=0))
            #print("tot len", len(buffer))
            buffer = []
            cnt += 1
            print(cnt, '/'+str(tot_ctx), file=sys.stderr)
    print("loaded ctx", file=sys.stderr)
    tot_query = encoded.shape[0]
    for i in range(tot_query//10):
        scores, idxes = index.search(encoded[i*10:(i+1)*10], 256)
        right = 0
        for j in range(10):
            for id in idxes[j]:
                if pctg[i*10+j] <= id < pctg[i*10+j]+min(lines[i*10+j]//2, 64) :
                    right += 1
            print(right/min(64, lines[i*10+j]//2))
            right = 0
            with open(save_path+file_list[i*10+j], mode='wb') as f:
                for idx in idxes[j]:
                    cnt, fidx = get_idx(idx, pctg)
                    seq_name = linecache.getline(src_list[cnt], 2*cnt+1)
                    seq_str  = linecache.getline(linecache.__file__, 2*cnt+2)
                    f.write(seq_name+seq_str)


