import pickle
import os
import numpy as np
from typing import Sequence, Tuple, List, Union
import esm
import torch
from torch.utils.data import Dataset, DataLoader
from model import MyEncoder
from data import EbdDataset, SingleConverter, DistributedProxySampler
from train import do_embedding
import faiss

BATCHSZ=8
path = "./test.txt"

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

if __name__ == "__main__":

    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    model = MyEncoder(encoder, 0)
    prev = torch.load('./full/19.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    device = torch.device("cuda:0")
    model = model.to(device)
    print("loaded model")

    batch_converter = SingleConverter(alphabet)
    dataset = EbdDataset(path)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=batch_converter)
    encoded = qencode(model, dataloader, device=device)
    print(encoded.shape)
    print("encoded query")

    ctx_list = os.listdir('./ebd')
    ctx_list.sort()
    #ctx_ebd = []
    buffer = []

    index = faiss.IndexFlatIP(768)
    cnt = 0
    for i in ctx_list:
        with open('./ebd/'+i, "rb") as reader:
            whole_doc = pickle.load(reader)
            for name, tok in whole_doc:
                buffer.append(tok)
            index.add(np.stack(buffer, axis=0))
            #print("tot len", len(buffer))
            buffer = []
            cnt += 1
            if cnt%100 == 0:
                print(cnt, 1000)
    print("loaded ctx")
    scores, idxes = index.search(encoded, 1024*10)
    right = 0
    for i in range(8):
        for id in idxes[i]:
            if i*1024 <= id < (i+1)*1024 :
                right += 1
        print(right/1024)
        right = 0

