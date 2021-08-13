import pickle
import wandb
import subprocess
import os
from typing import Sequence, Tuple, List, Union
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from model import MyEncoder
from data import MyDataset, BatchConverter, DistributedProxySampler
from train import train, evaluate

DISTRIBUTED = True
TRBATCHSZ = 16
EVBATCHSZ = 16
use_wandb = False
threshold = 0.7
eval_per_step = 40
lr = 1e-5
use_wandb = True
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"

def wc_count(file_name):
    return 200
    #out = subprocess.getoutput("wc -l %s" % file_name)
    #return int(out.split()[0])
    
def get_filename(path: str) -> List[str]:
    files = os.listdir(path)
    names = []
    lines = []
    for file in files:
        if ".a3m" in file:
            names.append(path + file)
    lines = [wc_count(name) for name in names]

    return names, lines

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
    print("loaded model")
    
    model = MyEncoder(encoder, 0)
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
    names, lines = get_filename(path)
    train_set = MyDataset(path, True, names, lines)
    eval_set = MyDataset(path, False, names, lines)
    trbatch = train_set.get_batch_indices(TRBATCHSZ)
    evbatch = train_set.get_batch_indices(EVBATCHSZ)
    if DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        trbatch = DistributedProxySampler(trbatch, num_replicas=4, rank=local_rank)
        evbatch = DistributedProxySampler(evbatch, num_replicas=4, rank=local_rank)
    
    batch_converter = BatchConverter(alphabet)
    train_loader = DataLoader(dataset=train_set, collate_fn=batch_converter, batch_sampler=trbatch)
    eval_loader = DataLoader(dataset=eval_set, collate_fn=batch_converter, batch_sampler=evbatch)
    print("loaded dataset")
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma = 0.85)
    train(
        model, 
        train_loader, 
        eval_loader, 
        n_epoches=20, 
        optimizer=optimizer, 
        threshold=threshold, 
        eval_per_step=eval_per_step, 
        use_wandb=use_wandb, 
        use_distr=DISTRIBUTED, 
        device=device, 
        acc_step=4
    )

