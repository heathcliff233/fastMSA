import os
import sys 
import re
import pickle
import math
from torch.utils import data
sys.path.append("/share/hongliang") 
import numpy as np
import pandas as pd
import phylopandas.phylopandas as ph
import numpy as np
import subprocess
from typing import Sequence, Tuple, List, Union
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from model import MyEncoder
from data import EbdDataset, SingleConverter, DistributedProxySampler

DISTRIBUTED = True
BATCHSZ = 128
save_per_step = 100
num_gpus = 4
#path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
path = "./c1000_msa/"
#save_path = './ur90_ebd/'
save_path = './t1000_ebd/'

class PdDataset(Dataset):
    def __init__(self, df, left, right):
        #self.records = []
        #for i in path:
        #    self.records.extend(list(SeqIO.parse(i, "fasta")))
        self.records = df.iloc[left:right]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records.iloc[index]
        #return rec.id, re.sub('[(a-z)(\-)]', '', rec.seq.__str__())
        return rec.id, re.sub('[(\-)]', '', rec.sequence)



def fasta_embedding(model, loader, set_seqs, save_dir, shift, use_distr=False, device="cuda:0"):
    model.eval()
    res = []
    ebd_seqs = 0
    for i, (ids, toks) in enumerate(loader):
        ebd_seqs += toks.shape[0]
        toks = toks.to(device)
        with torch.no_grad():
            if use_distr:
                out = model.module.forward_once(toks)
            else:
                out = model.forward_once(toks)
        out = out.cpu().numpy()
        res.extend(
            [(ids[j], out[j]) for j in range(out.shape[0])]
        )
        if (i+1)%save_per_step==0 or ebd_seqs==set_seqs:
            with open(save_dir+"ebd-%05d"%(shift+i//save_per_step), mode="wb") as f:
                pickle.dump(res, f)
            res = []
        


if __name__ == "__main__":
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    print("loaded model")
    
    model = MyEncoder(encoder, 0)
    prev = torch.load('./continue_train/59.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    #fasta_df = ph.read_fasta('/ssdcache/wangsheng/databases/uniref90/uniref90.fasta', use_uids=False)
    all_seqs_file = os.listdir(path)
    all_seqs_file.sort()
    src_list = [path+file for file in all_seqs_file]
    # Add phylopandas func
    df_list = [ph.read_fasta(src, use_uids=False) for src in src_list]
    fasta_df = pd.concat(df_list)
    tot_seqs = len(fasta_df)
    one_part = math.ceil(tot_seqs/num_gpus) 
    part_indexes = math.ceil(one_part/(BATCHSZ*save_per_step))
    device = torch.device("cuda:0")
    if DISTRIBUTED:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print("local rank ", local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        #model = nn.DataParallel(model, device_ids)

    model = model.to(device)

    if DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        dataset = PdDataset(fasta_df, local_rank*one_part, (local_rank+1)*one_part)
    else:
        dataset = PdDataset(fasta_df, 0, -1)
    
    batch_converter = SingleConverter(alphabet)
    set_seqs = len(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCHSZ, shuffle=False, collate_fn=batch_converter)

    fasta_embedding(model, dataloader, set_seqs, save_path, local_rank*part_indexes, use_distr=True, device=device)

