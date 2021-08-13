import os
import random
import torch
import tqdm
import wandb
path = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
from torch.utils.data import Dataset
from typing import Sequence, Tuple, List, Union
import linecache
import re
import subprocess
DISTRIBUTED = True
class  MyDataset(Dataset):
    def __init__(self, root: str, is_train: bool):
        self.root = root
        self.names, self.lines = self.get_filename(root)
        self.is_train = is_train

    def wc_count(self, file_name):
        #out = subprocess.getoutput("wc -l %s" % file_name)
        #return int(out.split()[0])
        return 200
    
    def get_filename(self, path: str) -> List[str]:
        files = os.listdir(path)
        names = []
        lines = []
        for file in files:
            if ".a3m" in file:
                names.append(path + file)
        lines = [self.wc_count(name) for name in names]

        return names, lines

    def get_pair(self, path: str, lines: int) -> Tuple[str, str]:
        if self.is_train:
            span = range(int(lines*0.8))
        else:
            span = range(int(lines*0.8,lines))
        idx1, idx2 = random.sample(span, 2)
        seq1 = re.sub('[(a-z)(-)]', '', linecache.getline(path, 2*idx1 + 2))
        seq2 = re.sub('[(a-z)(-)]', '', linecache.getline(path, 2*idx2 + 2))
        return seq1, seq2

    def __getitem__(self, index: int) -> Tuple[str, str]:
        seq1, seq2 = self.get_pair(self.names[index], self.lines[index])
        return seq1, seq2

    def __len__(self):
        return len(self.right - self.left)

    def get_batch_indices(self, batch_size: int) -> List[List[int]] :
        batches = []
        buf = []
        cnt = 0
        iters = len(self.names) // batch_size

        for i in range(iters):
            buf = random.sample(range(len(self.names)), batch_size)
            batches.append(buf)

        return batches


from typing import Sequence, Tuple, List, Union
class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max(max(len(seq1),len(seq2)) for seq1, seq2 in raw_batch)
        tokens1 = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens2 = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens1.fill_(self.alphabet.padding_idx)
        tokens2.fill_(self.alphabet.padding_idx)

        for i, (seq_str1, seq_str2) in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens1[i, 0] = self.alphabet.cls_idx
                tokens2[i, 0] = self.alphabet.cls_idx
            seq1 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str1], dtype=torch.int64)
            seq2 = torch.tensor([self.alphabet.get_idx(s) for s in seq_str2], dtype=torch.int64)
            tokens1[
                i,
                int(self.alphabet.prepend_bos) : len(seq_str1) + int(self.alphabet.prepend_bos),
            ] = seq1
            tokens2[
                i,
                int(self.alphabet.prepend_bos) : len(seq_str2) + int(self.alphabet.prepend_bos),
            ] = seq2
            if self.alphabet.append_eos:
                tokens1[i, len(seq_str1) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                tokens2[i, len(seq_str2) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return tokens1, tokens2


import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader

class MyEncoder(nn.Module):
    def __init__(self, bert, proj_dim):
        super(MyEncoder, self).__init__()
        self.bert = bert 
        self.num_layers = bert.num_layers
        repr_layers = -1
        self.repr_layers = (repr_layers + self.num_layers + 1) % (self.num_layers + 1)
        self.recast = nn.Linear(768, proj_dim) if proj_dim != 0 else None

    def forward_once(self, x):
        x = self.bert(x, repr_layers=[self.repr_layers])['representations'][self.repr_layers]
        x = x[:,0]
        #x = x.squeeze(1)
        if self.recast :
            x = self.recast(x)
        return x

    def forward(self, batch):
        seq1, seq2 = batch
        qebd = self.forward_once(seq1)
        cebd = self.forward_once(seq2)
        return qebd, cebd

    def get_loss(self, ebd):
        qebd, cebd = ebd 
        print("q/c vec shape :", qebd.shape, cebd.shape)
        sim_mx = dot_product_scores(qebd, cebd)
        #label = torch.eye(sim_mx.shape[0]).long()
        label = torch.arange(sim_mx.shape[0], dtype=torch.long)
        sm_score = F.log_softmax(sim_mx, dim=1)
        max_score, max_idxs = torch.max(sm_score, 1)
        correct_predictions_count = (
            max_idxs == label.to(sm_score.device)
        ).sum()
        print("prediction ", correct_predictions_count.detach().cpu().item(), sim_mx.shape[0])
        loss = F.nll_loss(
            sm_score,
            label.to(sm_score.device),
            reduction="mean"
        )
        return loss


def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def train(model, train_loader, eval_loader, n_epoches, optimizer, threshold=0.7, eval_per_step=10, use_wandb=False):
    if use_wandb:
        wandb.watch(model, log_freq=eval_per_step)
    for epoch in range(n_epoches):
        print("epoch " + str(epoch+1))
        cnt = 0
        tot_loss = 0
        
        model.train()
        print("train")
        #pbar = tqdm(train_loader, unit="batch")
        #for toks1, toks2 in pbar:
        for i, (toks1, toks2) in enumerate(train_loader):
            toks1, toks2 = toks1.cuda(non_blocking=True), toks2.cuda(non_blocking=True)
            #print(toks1[:5])
            print("data shape ", toks1.shape, toks2.shape)
            cnt += 1
            
            if cnt%eval_per_step == 0 :
                if cnt%(eval_per_step*1)==0:
                    acc = evaluate(model, eval_loader, threshold)
                    ac2 = evaluate(model, train_loader,threshold)
                    print("loss"+str(tot_loss/eval_per_step))
                    if use_wandb :
                        wandb.log({"train/train-acc": ac2, "train/eval-acc": acc,"train/loss": tot_loss/eval_per_step})
                    tot_loss = 0
                model.train()
            
            optimizer.zero_grad()
            if DISTRIBUTED:
                loss = model.module.get_loss(model((toks1, toks2)))
            else:
                loss = model.get_loss(model((toks1, toks2)))
            '''
            with torch.no_grad():
                p, n = model.get_avg(model.forward_once(inputs), labels)
            #print(p, n)
            '''
            tot_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()
        #save(model, epoch)


def evaluate(model, loader, threshold):
    return 0


import esm

BATCHSZ = 64
device_ids = [0, 1, 2, 3]
use_wandb = False
threshold = 0.7
eval_per_step = 40
lr = 1e-5
if __name__ == "__main__":
    encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    #print(alphabet.__dict__)
    print("loaded model")
    model = MyEncoder(encoder, 0)
    if DISTRIBUTED:
        model = nn.DataParallel(model, device_ids)
    model = model.cuda()
    train_set = MyDataset(path, True)
    eval_set = MyDataset(path, False)
    trbatch = train_set.get_batch_indices(BATCHSZ)
    evbatch = eval_set.get_batch_indices(BATCHSZ)
    batch_converter = BatchConverter(alphabet)
    train_loader = DataLoader(dataset=train_set, collate_fn=batch_converter, batch_sampler=trbatch)
    eval_loader = DataLoader(dataset=eval_set, collate_fn=batch_converter, batch_sampler=evbatch)
    print("loaded dataset")
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma = 0.85)
    train(model, train_loader, eval_loader, n_epoches=20, optimizer=optimizer, threshold=threshold, eval_per_step=eval_per_step, use_wandb=use_wandb)

