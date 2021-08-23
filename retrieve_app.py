import pickle
import os
import sys
import re
import numpy as np
import esm
import torch
from torch.utils.data import Dataset, DataLoader
from model import MyEncoder
from data import QueryDataset, SingleConverter
from myutils import get_filename
import faiss
import linecache
import streamlit as st

BATCHSZ=8
tar_num = 2048
path = "./split/split_dataset_test.txt"
qdir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_seq/"
msadir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
ctx_dir = "./split_ebd/"
save_path = "./pred-202108232011-2048/"

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
    while idx >= lst[cnt] :
        cnt += 1
        if cnt == tot :
            break
    cnt -= 1
    return cnt, idx-lst[cnt]


@st.cache
def get_model():
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    model = MyEncoder(encoder, 0)
    prev = torch.load('./split_train/39.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    batch_converter = SingleConverter(alphabet)

    return model, batch_converter

@st.cache
def gen_ctx_ebd():
    path_list = np.genfromtxt(path, dtype='str').T[0]
    path_list.sort()

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
            #print('\r ', cnt, '/'+str(tot_ctx), file=sys.stderr, end="")
    #print("loaded ctx", file=sys.stderr)
    return index


model, batch_converter = get_model()
device = torch.device("cuda:0")
model = model.to(device)
index = gen_ctx_ebd()


if __name__ == "__main__":

    dataset = QueryDataset(query_list)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=batch_converter)
    encoded = qencode(model, dataloader, device=device)
    print(encoded.shape, file=sys.stderr)
    print("encoded query", file=sys.stderr)
    
    
    
    #print(pctg[:10])
    tot_query = encoded.shape[0]
    for i in range(tot_query//10):
        scores, idxes = index.search(encoded[i*10:(i+1)*10], tar_num)
        right = 0
        for j in range(10):
            for id in idxes[j]:
                if pctg[i*10+j] <= id < pctg[i*10+j]+lines[i*10+j]//2 :
                    right += 1
            print(file_list[i*10+j], right, min(tar_num, lines[i*10+j]//2))
            right = 0
            f = open(save_path+file_list[i*10+j], mode='w')
            #print(pctg[i*10+j])
            #print(idxes[j])
            for idx in idxes[j]:
                cnt, fidx = get_idx(idx, pctg)
                seq_name = linecache.getline(src_list[cnt], 2*fidx+1)
                #seq_str  = re.sub('[(a-z)(\-)]', '', linecache.getline(src_list[cnt], 2*fidx+2))
                seq_str  = re.sub('[(\-)]', '', linecache.getline(src_list[cnt], 2*fidx+2))
                f.write(str(seq_name)+str(seq_str))
            f.close()
        print("\r Finished ", i, '%', file=sys.stderr, end="")


