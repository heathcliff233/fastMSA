import os
import sys
sys.path.append('/share/hongliang/')
import numpy as np
import pandas as pd
import phylopandas.phylopandas as ph
import subprocess
from typing import Sequence, Tuple, List, Union
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
sys.path.append('/user/hongliang/mydpr')
from model.model import MyEncoder
from data.data import PdDataset, SingleConverter, DistributedProxySampler
from train.train import do_embedding
from utils.myutils import wc_count, get_filename

DISTRIBUTED = True
BATCHSZ = 128
num_gpus = 4
#path = "./testset/"
#fasta_path = '/share/hongliang/seq_db.fasta'
#fasta_path = '/share/hongliang/res-database.fasta'
fasta_path = '/ssdcache/wangsheng/databases/uniref90/uniref90.fasta'
#fasta_path = '/ssdcache/zhengliangzhen/hongliang/ur90_201803.fasta'
#save_path = './random_ebd/'
save_path = './fseq_ebd_v2/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,4,5"

if __name__ == "__main__":
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    print("loaded model")
    
    model = MyEncoder(encoder, 0)
    prev = torch.load('./model_from_dgx/v2/99.pth')
    #prev = torch.load('./continue_train/59.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    #model.load_state_dict(torch.load('./full/19.pth').module)
    device = torch.device("cuda:0")
    local_rank = 0
    if DISTRIBUTED:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print("local rank ", local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        #model = nn.DataParallel(model, device_ids)

    model = model.to(device)

    df = ph.read_fasta(fasta_path)
    df_for_this_node = df
    if DISTRIBUTED:
        len_df = df.shape[0]
        df_for_this_node = df.iloc[local_rank*len_df//num_gpus : (local_rank+1)*len_df//num_gpus]
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    batch_converter = SingleConverter(alphabet)
    lsdf = df_for_this_node.shape[0]
    df_list = [df_for_this_node.iloc[i*(lsdf//100+1):(i+1)*(lsdf//100+1)] for i in range(100)]
    for i in range(100) :
        dlog = df_list[i]
        dataset = PdDataset(dlog)
        dataloader = DataLoader(dataset=dataset, batch_size=BATCHSZ, shuffle=False, collate_fn=batch_converter)
        name = save_path+"ebd-%02d-%03d----"%(local_rank, i)
        do_embedding(model, dataloader, name, use_distr=DISTRIBUTED, device=device)
        #if (DISTRIBUTED and torch.distributed.get_rank()==0) or (not DISTRIBUTED):
        print("finished embedding dataset ", name[:-4])

