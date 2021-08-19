import os
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
from train import do_embedding
from myutils import wc_count, get_filename

DISTRIBUTED = True
BATCHSZ = 128
#path = "./testset/"
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
save_path = './split_ebd/'

if __name__ == "__main__":
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    #encoder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    #encoder.load_state_dict(torch.load('./full/19.pth'))
    print("loaded model")
    
    model = MyEncoder(encoder, 0)
    prev = torch.load('./split_train/39.pth')
    later = dict((k[7:], v) for (k,v) in prev.items())
    model.load_state_dict(later)
    #model.load_state_dict(torch.load('./full/19.pth').module)
    device = torch.device("cuda:0")
    if DISTRIBUTED:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print("local rank ", local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        #model = nn.DataParallel(model, device_ids)

    model = model.to(device)

    #file_list = os.listdir(path)
    ttpath = './split/split_dataset_test.txt'
    file_list, lines = get_filename(ttpath)
    
    #sorted_id = sorted(range(len(file_list)), key=lambda k: file_list[k])
    file_list.sort()
    #lines = [lines[i] for i in sorted_id]
    #pctg = [0]*len(lines)
    #for i in range(1, len(lines)):
    #    pctg[i] = pctg[i-1] + lines[i-1]
    file_for_this_node = file_list
    if DISTRIBUTED:
        file_for_this_node = file_list[local_rank::4]
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    batch_converter = SingleConverter(alphabet)
    for name in file_for_this_node :
        dataset = EbdDataset(path+name)
        dataloader = DataLoader(dataset=dataset, batch_size=BATCHSZ, shuffle=False, collate_fn=batch_converter)
        #if (DISTRIBUTED and torch.distributed.get_rank()==0) or (not DISTRIBUTED):
        #print("loaded dataset ", name)
        do_embedding(model, dataloader, save_path+name, use_distr=DISTRIBUTED, device=device)
        #if (DISTRIBUTED and torch.distributed.get_rank()==0) or (not DISTRIBUTED):
        print("finished embedding dataset ", name)

