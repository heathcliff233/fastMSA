import pickle
import wandb
import subprocess
import os
import sys
from typing import Sequence, Tuple, List, Union
import numpy as np
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
sys.path.append('/user/hongliang/mydpr')
from model.model import MyEncoder
from data.data import MyDataset, BatchConverter, DistributedProxySampler
from train.train import train, evaluate
from utils.myutils import get_filename

DISTRIBUTED = True
TRBATCHSZ = 24
EVBATCHSZ = 64
threshold = 0.7
eval_per_step = 30
lr = 1e-6
use_wandb = True
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"

#def wc_count(file_name):
    #return 4
    #out = subprocess.getoutput("wc -l %s" % file_name)
    #res = int(out.split()[0])
    #return res
    
#def get_filename(sel_path: str) -> List[str]:
    #path_list = np.genfromtxt(sel_path, dtype='str').T[0]
    #names = [path+str(name)+'.a3m' for name in path_list]
    #lines = [wc_count(name) for name in names]
    #return names, lines

def init_wandb():
    wandb.init(
        project="Retrieval",
        config= {
            "optim" : "AdamW",
            "lr" : lr,
            "train_batch" : TRBATCHSZ,
            "eval_per_step" : EVBATCHSZ,
        }
    )


if __name__ == "__main__":
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    #encoder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    print("loaded model")
    
    model = MyEncoder(encoder, 0)
    ##################################
    prev = torch.load('./continue_train/59.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    ##################################
    device = torch.device("cuda:0")
    if DISTRIBUTED:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print("local rank ", local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        #model = nn.DataParallel(model, device_ids)

    if use_wandb:
        if DISTRIBUTED:
            if torch.distributed.get_rank()==0:
                init_wandb()
        else:
            init_wandb()
    
    model = model.to(device)
    trpath = './split/train.txt'
    trnames, trlines = get_filename(trpath)
    trnames = [path+name for name in trnames]
    evpath = './split/valid.txt'
    evnames, evlines = get_filename(evpath)
    evnames = [path+name for name in evnames]
    print(len(trnames))
    train_set = MyDataset(trnames, trlines)
    eval_set = MyDataset(evnames, evlines)
    trbatch = train_set.get_batch_indices(TRBATCHSZ)
    evbatch = eval_set.get_batch_indices(EVBATCHSZ)
    if DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        trbatch = DistributedProxySampler(trbatch, num_replicas=dist.get_world_size(), rank=local_rank)
        evbatch = DistributedProxySampler(evbatch, num_replicas=dist.get_world_size(), rank=local_rank)
    
    batch_converter = BatchConverter(alphabet)
    train_loader = DataLoader(dataset=train_set, collate_fn=batch_converter, batch_sampler=trbatch)
    eval_loader = DataLoader(dataset=eval_set, collate_fn=batch_converter, batch_sampler=evbatch)
    if (DISTRIBUTED and torch.distributed.get_rank()==0) or (not DISTRIBUTED):
        print("loaded dataset")
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma = 0.85)
    print("res: ", evaluate(model, eval_loader, use_distr=True))
    
    train(
        model, 
        train_loader, 
        eval_loader, 
        n_epoches=60, 
        optimizer=optimizer, 
        threshold=threshold, 
        eval_per_step=eval_per_step, 
        use_wandb=use_wandb, 
        use_distr=DISTRIBUTED, 
        device=device, 
        acc_step=4
    )
    

